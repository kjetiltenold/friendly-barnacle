"""Tool definitions for OpenAI tool-use and dispatch to Tripletex API."""

import datetime
import json
import logging
import re
import uuid
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from app.config import get_settings
from app.endpoint_search import EndpointSearchClient
from app.tripletex.client import TripletexClient

logger = logging.getLogger(__name__)

PASSIVE_SEARCH_PARAMS = {
    "fields",
    "from",
    "count",
    "sorting",
    "invoiceDateFrom",
    "invoiceDateTo",
    "dateFrom",
    "dateTo",
    "startDateFrom",
    "startDateTo",
    "endDateFrom",
    "endDateTo",
    "returnDateFrom",
    "returnDateTo",
    "departureDateFrom",
    "departureDateTo",
    "periodStart",
    "periodEnd",
}


@dataclass
class EntityContext:
    """Tracks created entity IDs so we can auto-inject missing references.

    GPT-4o frequently omits required entity references (customer on orders,
    orders on invoices) even when told to include them.  This safety net
    fills in the most-recently-created entity when the model forgets.
    """

    last_customer_id: int | None = None
    last_product_id: int | None = None
    product_ids: list[int] | None = None  # All product IDs created/found
    last_order_id: int | None = None
    last_employee_id: int | None = None
    last_project_id: int | None = None
    last_invoice_id: int | None = None
    last_travel_expense_id: int | None = None
    last_activity_id: int | None = None
    last_rate_category_id: int | None = None
    last_cost_category_id: int | None = None
    last_payment_type_id: int | None = None
    last_hourly_rate_id: int | None = None
    last_dimension_index: int | None = None
    last_dimension_value_id: int | None = None

    def __post_init__(self):
        if self.product_ids is None:
            self.product_ids = []

    def track(self, name: str, result: dict) -> None:
        """Extract and store the entity ID from a creation response."""
        value = result.get("value", {})
        entity_id = value.get("id")
        if entity_id is None:
            return
        mapping = {
            "create_customer": "last_customer_id",
            "create_product": "last_product_id",
            "create_order": "last_order_id",
            "create_employee": "last_employee_id",
            "create_project": "last_project_id",
            "create_invoice": "last_invoice_id",
            "create_travel_expense": "last_travel_expense_id",
        }
        attr = mapping.get(name)
        if attr:
            setattr(self, attr, entity_id)
            logger.info(f"EntityContext: {attr} = {entity_id}")
        # Track all product IDs for multi-product orders
        if name == "create_product" and entity_id not in self.product_ids:
            self.product_ids.append(entity_id)
        if name == "create_accounting_dimension_name":
            dimension_index = value.get("dimensionIndex")
            if dimension_index is not None:
                self.last_dimension_index = dimension_index
        if name == "create_accounting_dimension_value":
            self.last_dimension_value_id = entity_id


def _track_lookup_context(ctx: EntityContext | None, path: str, result: dict) -> None:
    """Capture useful IDs from raw GET lookups for later tool auto-fill."""
    if ctx is None:
        return
    values = result.get("values", [])
    if not values:
        return
    first = values[0]
    first_id = first.get("id")
    if first_id is None:
        return

    if path.startswith("/activity"):
        ctx.last_activity_id = first_id
    elif path.startswith("/travelExpense/rateCategory"):
        ctx.last_rate_category_id = first_id
    elif path.startswith("/travelExpense/costCategory"):
        ctx.last_cost_category_id = first_id
    elif path.startswith("/travelExpense/paymentType") or path.startswith("/invoice/paymentType"):
        ctx.last_payment_type_id = first_id
    elif path.startswith("/project/hourlyRates"):
        ctx.last_hourly_rate_id = first_id


def _tool(name: str, description: str, parameters: dict) -> dict:
    """Wrap a function tool in OpenAI format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


BASE_TOOL_DEFINITIONS = [
    _tool("create_employee", "Create a new employee in Tripletex.", {
        "type": "object",
        "properties": {
            "firstName": {"type": "string"},
            "lastName": {"type": "string"},
            "email": {"type": "string"},
            "dateOfBirth": {"type": "string", "description": "YYYY-MM-DD"},
            "phoneNumberMobileCountryCode": {"type": "string", "description": "Country code, e.g. +47"},
            "phoneNumberMobile": {"type": "string"},
            "userType": {"type": "string", "enum": ["STANDARD", "EXTENDED", "NO_ACCESS"], "description": "User access level. Defaults to STANDARD. Use EXTENDED for admin roles."},
            "startDate": {"type": "string", "description": "Employment start date YYYY-MM-DD. Creates an employment record for the employee."},
        },
        "required": ["firstName", "lastName"],
    }),
    _tool("update_employee", "Update an existing employee by ID.", {
        "type": "object",
        "properties": {
            "employee_id": {"type": "integer"},
            "fields": {"type": "object", "description": "Fields to update (firstName, lastName, email, etc.)"},
        },
        "required": ["employee_id", "fields"],
    }),
    _tool("create_customer", "Create a new customer or supplier in Tripletex.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phoneNumber": {"type": "string"},
            "organizationNumber": {"type": "string"},
            "isCustomer": {"type": "boolean", "default": True},
            "isSupplier": {"type": "boolean"},
            "postalAddress": {
                "type": "object",
                "description": "Postal/physical address",
                "properties": {
                    "addressLine1": {"type": "string", "description": "Street address, e.g. Kirkegata 132"},
                    "postalCode": {"type": "string", "description": "e.g. 7010"},
                    "city": {"type": "string", "description": "e.g. Trondheim"},
                },
            },
        },
        "required": ["name"],
    }),
    _tool("update_customer", "Update an existing customer by ID.", {
        "type": "object",
        "properties": {
            "customer_id": {"type": "integer"},
            "fields": {"type": "object", "description": "Fields to update"},
        },
        "required": ["customer_id", "fields"],
    }),
    _tool("create_product", "Create a new product in Tripletex.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "number": {"type": "string", "description": "Product number/SKU"},
            "costExcludingVatCurrency": {"type": "number"},
            "priceExcludingVatCurrency": {"type": "number"},
            "priceIncludingVatCurrency": {"type": "number"},
            "vatType": {"type": "object", "description": "VAT type object, e.g. {\"id\": 3} for 25% MVA"},
        },
        "required": ["name"],
    }),
    _tool("create_order", "Create a sales order. MUST include customer, product references, and vatType on EVERY order line.", {
        "type": "object",
        "properties": {
            "customer": {"type": "object", "description": "REQUIRED — customer reference object, e.g. {\"id\": 123} using the id from create_customer response"},
            "project": {"type": "object", "description": "Optional project reference object, e.g. {\"id\": 456}, for project-linked orders and invoices."},
            "orderDate": {"type": "string", "description": "YYYY-MM-DD"},
            "deliveryDate": {"type": "string", "description": "YYYY-MM-DD"},
            "isPrioritizeAmountsIncludingVat": {"type": "boolean", "description": "False when using unitPriceExcludingVatCurrency, true when using unitPriceIncludingVatCurrency."},
            "orderLines": {
                "type": "array",
                "description": "EVERY line MUST have product ref AND vatType ref. Look up vatType IDs from GET /ledger/vatType first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "object", "description": "REQUIRED — Product reference {\"id\": product_id} from create_product response"},
                        "description": {"type": "string"},
                        "count": {"type": "number"},
                        "unitPriceExcludingVatCurrency": {"type": "number"},
                        "unitPriceIncludingVatCurrency": {"type": "number"},
                        "vatType": {"type": "object"},
                    },
                    "required": ["product", "count", "vatType"],
                },
            },
        },
        "required": ["customer", "orderDate"],
    }),
    _tool("create_invoice", "Create an invoice from an existing order. MUST include the orders array referencing order IDs from create_order.", {
        "type": "object",
        "properties": {
            "invoiceDate": {"type": "string", "description": "YYYY-MM-DD"},
            "invoiceDueDate": {"type": "string", "description": "YYYY-MM-DD"},
            "orders": {
                "type": "array",
                "description": "REQUIRED — array of order references, e.g. [{\"id\": 456}] using the id from create_order response",
                "items": {"type": "object", "description": "Order reference {\"id\": order_id}"},
            },
        },
        "required": ["invoiceDate", "invoiceDueDate", "orders"],
    }),
    _tool("create_project", "Create a project in Tripletex.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "number": {"type": "string"},
            "projectManager": {"type": "object", "description": "{\"id\": employee_id}"},
            "customer": {"type": "object", "description": "{\"id\": customer_id}"},
            "startDate": {"type": "string"},
            "endDate": {"type": "string"},
            "isClosed": {"type": "boolean"},
        },
        "required": ["name", "number", "projectManager"],
    }),
    _tool("create_department", "Create a department in Tripletex.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "departmentNumber": {"type": "string"},
        },
        "required": ["name"],
    }),
    _tool("create_travel_expense", "Create a travel expense report in Tripletex.", {
        "type": "object",
        "properties": {
            "employee": {"type": "object", "description": "{\"id\": employee_id}"},
            "project": {"type": "object", "description": "{\"id\": project_id}"},
            "title": {"type": "string"},
            "departureDate": {"type": "string", "description": "YYYY-MM-DD (auto-nested into travelDetails)"},
            "returnDate": {"type": "string", "description": "YYYY-MM-DD (auto-nested into travelDetails)"},
            "travelDetails": {"type": "object", "description": "Travel details: departureDate, returnDate, destination, purpose, isDayTrip, isForeignTravel"},
            "perDiemCompensations": {"type": "array", "items": {"type": "object"}, "description": "Per diem compensations array"},
            "costs": {"type": "array", "items": {"type": "object"}, "description": "Cost items array"},
        },
        "required": ["employee", "title"],
    }),
    _tool("create_per_diem_compensation", "Create a travel expense per diem compensation. MUST include travelExpense, rateCategory, location, overnightAccommodation, count, and rate.", {
        "type": "object",
        "properties": {
            "travelExpense": {"type": "object", "description": "{\"id\": travel_expense_id}"},
            "rateCategory": {"type": "object", "description": "{\"id\": rate_category_id}"},
            "location": {"type": "string"},
            "overnightAccommodation": {
                "type": "string",
                "enum": ["NONE", "HOTEL", "BOARDING_HOUSE_WITHOUT_COOKING", "BOARDING_HOUSE_WITH_COOKING"],
            },
            "count": {"type": "integer"},
            "rate": {"type": "number"},
            "countryCode": {"type": "string"},
            "address": {"type": "string"},
        },
        "required": ["travelExpense", "rateCategory", "location", "overnightAccommodation", "count", "rate"],
    }),
    _tool("create_travel_cost", "Create a travel expense cost line. MUST include travelExpense, costCategory, comments, amountCurrencyIncVat, and date.", {
        "type": "object",
        "properties": {
            "travelExpense": {"type": "object", "description": "{\"id\": travel_expense_id}"},
            "costCategory": {"type": "object", "description": "{\"id\": cost_category_id}"},
            "paymentType": {"type": "object", "description": "{\"id\": payment_type_id}"},
            "comments": {"type": "string"},
            "amountCurrencyIncVat": {"type": "number"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "vatType": {"type": "object"},
            "currency": {"type": "object"},
            "rate": {"type": "number"},
        },
        "required": ["travelExpense", "costCategory", "comments", "amountCurrencyIncVat", "date"],
    }),
    _tool("create_project_activity", "Link an activity to a project. MUST include both project and activity references.", {
        "type": "object",
        "properties": {
            "project": {"type": "object", "description": "{\"id\": project_id}"},
            "activity": {"type": "object", "description": "{\"id\": activity_id}"},
        },
        "required": ["project", "activity"],
    }),
    _tool("create_timesheet_entry", "Create a timesheet entry. MUST include employee, project, activity, date, and hours.", {
        "type": "object",
        "properties": {
            "employee": {"type": "object", "description": "{\"id\": employee_id}"},
            "project": {"type": "object", "description": "{\"id\": project_id}"},
            "activity": {"type": "object", "description": "{\"id\": activity_id}"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "hours": {"type": "number"},
        },
        "required": ["employee", "project", "activity", "date", "hours"],
    }),
    _tool("update_project_hourly_rate", "Update a project hourly rate. MUST include hourly_rate_id and fixedRate. Include project, startDate, and showInProjectOrder when known.", {
        "type": "object",
        "properties": {
            "hourly_rate_id": {"type": "integer"},
            "project": {"type": "object", "description": "{\"id\": project_id}"},
            "startDate": {"type": "string", "description": "YYYY-MM-DD"},
            "hourlyRateModel": {"type": "string"},
            "fixedRate": {"type": "number"},
            "showInProjectOrder": {"type": "boolean"},
        },
        "required": ["hourly_rate_id", "fixedRate"],
    }),
    _tool("create_accounting_dimension_name", "Create a free accounting dimension name. MUST include dimensionName.", {
        "type": "object",
        "properties": {
            "dimensionName": {"type": "string"},
            "description": {"type": "string"},
            "active": {"type": "boolean"},
        },
        "required": ["dimensionName"],
    }),
    _tool("create_accounting_dimension_value", "Create a free accounting dimension value. MUST include displayName and dimensionIndex.", {
        "type": "object",
        "properties": {
            "displayName": {"type": "string"},
            "dimensionIndex": {"type": "integer"},
            "active": {"type": "boolean"},
            "number": {"type": "string"},
            "showInVoucherRegistration": {"type": "boolean"},
        },
        "required": ["displayName", "dimensionIndex"],
    }),
    _tool("create_voucher", "Create a voucher with postings. MUST include date, description, and a postings array. Each posting must include account and amountGross.", {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "description": {"type": "string"},
            "postings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "account": {"type": "object", "description": "{\"id\": account_id}"},
                        "amountGross": {"type": "number", "description": "Positive = debit, negative = credit"},
                        "amountGrossCurrency": {"type": "number"},
                        "supplier": {"type": "object", "description": "{\"id\": supplier_id}"},
                        "customer": {"type": "object", "description": "{\"id\": customer_id}"},
                        "employee": {"type": "object", "description": "{\"id\": employee_id}"},
                        "project": {"type": "object", "description": "{\"id\": project_id}"},
                        "product": {"type": "object", "description": "{\"id\": product_id}"},
                        "vatType": {"type": "object", "description": "{\"id\": vat_type_id}"},
                        "description": {"type": "string"},
                        "freeAccountingDimension1": {"type": "object", "description": "{\"id\": dimension_value_id}"},
                        "freeAccountingDimension2": {"type": "object", "description": "{\"id\": dimension_value_id}"},
                        "freeAccountingDimension3": {"type": "object", "description": "{\"id\": dimension_value_id}"},
                    },
                    "required": ["account", "amountGross"],
                },
            },
            "year": {"type": "integer"},
        },
        "required": ["date", "description", "postings"],
    }),
    _tool("create_salary_transaction", "Create a salary transaction. MUST include date, year, month, and payslips. Each payslip must include employee and specifications. Each specification must include salaryType, rate, and count.", {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "year": {"type": "integer"},
            "month": {"type": "integer"},
            "payslips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "employee": {"type": "object", "description": "{\"id\": employee_id}"},
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                        "year": {"type": "integer"},
                        "month": {"type": "integer"},
                        "specifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "salaryType": {"type": "object", "description": "{\"id\": salary_type_id}"},
                                    "rate": {"type": "number"},
                                    "count": {"type": "number"},
                                    "description": {"type": "string"},
                                    "project": {"type": "object", "description": "{\"id\": project_id}"},
                                    "department": {"type": "object", "description": "{\"id\": department_id}"},
                                },
                                "required": ["salaryType", "rate", "count"],
                            },
                        },
                    },
                    "required": ["employee", "specifications"],
                },
            },
            "isHistorical": {"type": "boolean"},
            "paySlipsAvailableDate": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["date", "year", "month", "payslips"],
    }),
    _tool("search_entity", "Search for entities in Tripletex. Use sparingly — the account starts empty so searches on a fresh account return nothing.", {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "description": "API entity name: employee, customer, product, invoice, order, project, department, travelExpense",
            },
            "params": {
                "type": "object",
                "description": "Query parameters (e.g. {\"name\": \"Acme\", \"fields\": \"id,name\", \"count\": 10})",
            },
        },
        "required": ["entity_type"],
    }),
    _tool("get_entity", "Get a specific entity by type and ID.", {
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "entity_id": {"type": "integer"},
        },
        "required": ["entity_type", "entity_id"],
    }),
    _tool("delete_entity", "Delete an entity by type and ID.", {
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "entity_id": {"type": "integer"},
        },
        "required": ["entity_type", "entity_id"],
    }),
    _tool("tripletex_api_call", "Make a raw Tripletex API call. Use this as a fallback for operations not covered by other tools.", {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
            "path": {"type": "string", "description": "API path, e.g. /ledger/voucher"},
            "params": {"type": "object", "description": "Query parameters (for GET)"},
            "body": {"type": "object", "description": "JSON body (for POST/PUT)"},
        },
        "required": ["method", "path"],
    }),
]

ENDPOINT_SEARCH_TOOL = _tool(
    "find_tripletex_endpoints",
    "Search the indexed Tripletex endpoint catalog for the best-matching raw API endpoints before using tripletex_api_call.",
    {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Describe the action you need to perform, including the entity and the intended outcome.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "description": "Optional HTTP method to narrow the search.",
            },
            "top_k": {
                "type": "integer",
                "description": "Optional number of matches to return. Defaults to the configured endpoint search result count.",
            },
        },
        "required": ["task"],
    },
)


def get_tool_definitions() -> list[dict]:
    definitions = list(BASE_TOOL_DEFINITIONS)
    if get_settings().azure_search_configured:
        definitions.append(ENDPOINT_SEARCH_TOOL)
    return definitions


def _compact_dict(payload: dict) -> dict:
    """Drop empty values before sending payloads to Tripletex."""
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def _rewrite_fields_filter(fields: str, replacements: dict[str, str]) -> str:
    """Rewrite invalid field aliases while preserving order."""
    rewritten: list[str] = []
    seen: set[str] = set()
    for raw_part in fields.split(","):
        part = raw_part.strip()
        if not part:
            continue
        normalized = replacements.get(part, part)
        if normalized in seen:
            continue
        seen.add(normalized)
        rewritten.append(normalized)
    return ",".join(rewritten)


def _has_meaningful_search_filters(params: dict) -> bool:
    """Treat unfiltered list requests as unsafe to avoid random matches."""
    for key, value in params.items():
        if key in PASSIVE_SEARCH_PARAMS:
            continue
        if value in (None, "", [], {}):
            continue
        return True
    return False


def _generate_project_number() -> str:
    """Generate a unique-enough project number for generic prompts."""
    return f"P-{datetime.date.today():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def _is_placeholder_project_number(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().upper()
    return normalized in {"", "1", "P1", "P01", "P001", "PROJECT", "PRJ", "DEFAULT"}


def _is_duplicate_error(exc: Exception, *needles: str) -> bool:
    message = str(exc).lower()
    return any(needle.lower() in message for needle in needles)


async def dispatch_tool(
    client: TripletexClient,
    name: str,
    args_json: str,
    endpoint_search: EndpointSearchClient | None = None,
    ctx: EntityContext | None = None,
) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        args = json.loads(args_json)
        logger.info(f"Tool {name} args: {json.dumps(args, default=str, ensure_ascii=False)}")
        result = await _execute(client, name, args, endpoint_search=endpoint_search, ctx=ctx)
        if ctx is not None:
            ctx.track(name, result)
        # Log creation responses for scoring diagnostics
        if name.startswith("create_"):
            result_str = json.dumps(result, default=str, ensure_ascii=False)
            logger.info(f"Tool {name} response: {result_str[:500]}")
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Tool {name} failed: {e}")
        return json.dumps({"error": str(e)})


async def _ensure_department(client: TripletexClient) -> int | None:
    """Find or create a default department for employee creation."""
    try:
        result = await client.get("/department", params={"fields": "id,name", "count": 1})
        values = result.get("values", [])
        if values:
            logger.info(f"Found existing department id={values[0]['id']}")
            return values[0]["id"]
    except Exception:
        pass
    try:
        result = await client.post("/department", json={"name": "Avdeling", "departmentNumber": "1"})
        dept_id = result.get("value", {}).get("id")
        logger.info(f"Created default department id={dept_id}")
        return dept_id
    except Exception as e:
        logger.warning(f"Failed to create department: {e}")
        return None


async def _ensure_bank_account(client: TripletexClient) -> None:
    """Register a bank account so invoices can be created.

    Uses the ledger account system — find a bank account in the chart
    of accounts and ensure it has a bank account number set.
    """
    # Approach 1: Find an existing bank account in the chart of accounts
    try:
        result = await client.get("/ledger/account", params={
            "fields": "id,number,name,isBankAccount,bankAccountNumber",
            "isBankAccount": "true",
            "count": 5,
        })
        values = result.get("values", [])
        # Find one without a bank account number, or use any bank account
        for acct in values:
            if acct.get("isBankAccount") and not acct.get("bankAccountNumber"):
                acct_id = acct["id"]
                logger.info(f"Found bank account id={acct_id} number={acct.get('number')} without bank account number")
                await client.put(f"/ledger/account/{acct_id}", json={
                    "id": acct_id,
                    "number": acct.get("number"),
                    "name": acct.get("name", "Bank"),
                    "isBankAccount": True,
                    "bankAccountNumber": "12345678903",
                })
                logger.info(f"Set bankAccountNumber on ledger account id={acct_id}")
                return
        if values:
            logger.info("Bank accounts found but all have numbers set already")
            return
    except Exception as e:
        logger.info(f"Bank account lookup failed: {e}")

    # Approach 2: Find account 1920 (standard bank account) and set it up
    try:
        result = await client.get("/ledger/account", params={
            "number": "1920",
            "fields": "id,number,name,isBankAccount,bankAccountNumber",
        })
        values = result.get("values", [])
        if values:
            acct = values[0]
            acct_id = acct["id"]
            await client.put(f"/ledger/account/{acct_id}", json={
                "id": acct_id,
                "number": acct.get("number", 1920),
                "name": acct.get("name", "Bank"),
                "isBankAccount": True,
                "bankAccountNumber": "12345678903",
            })
            logger.info(f"Set bankAccountNumber on account 1920 (id={acct_id})")
            return
    except Exception as e:
        logger.warning(f"Failed to set bank account on 1920: {e}")


async def _execute(
    client: TripletexClient,
    name: str,
    args: dict,
    endpoint_search: EndpointSearchClient | None,
    ctx: EntityContext | None = None,
) -> dict:
    if name == "create_employee":
        if "userType" not in args:
            args["userType"] = "STANDARD"
        # Move startDate into an employments array if provided
        start_date = args.pop("startDate", None)
        if start_date and "employments" not in args:
            args["employments"] = [{"startDate": start_date}]
        # Strip empty/invalid nested objects that cause validation errors
        for ref_field in ("department", "employeeCategory"):
            val = args.get(ref_field)
            if isinstance(val, dict) and not val.get("id"):
                del args[ref_field]
        email = args.get("email")
        # Search first if email given — avoids 422 error on conflict
        if email:
            try:
                result = await client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email"})
                values = result.get("values", [])
                if values:
                    employee = values[0]
                    employee_id = employee["id"]
                    logger.info(f"Reusing existing employee id={employee_id} for email {email}")
                    return {"value": employee}
            except Exception:
                pass
        try:
            return await client.post("/employee", json=args)
        except Exception as e:
            # Auto-inject department if required
            if "department" in str(e) and "department" not in args:
                logger.info("Employee requires department — auto-finding/creating one")
                dept_id = await _ensure_department(client)
                if dept_id:
                    args["department"] = {"id": dept_id}
                    try:
                        return await client.post("/employee", json=args)
                    except Exception as retry_error:
                        if email and _is_duplicate_error(retry_error, "bruker med denne e-postadressen", "already exists"):
                            result = await client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email"})
                            values = result.get("values", [])
                            if values:
                                employee = values[0]
                                logger.info(f"Employee create hit duplicate email; reusing id={employee['id']}")
                                return {"value": employee}
                        raise
            if email and _is_duplicate_error(e, "bruker med denne e-postadressen", "already exists"):
                result = await client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email"})
                values = result.get("values", [])
                if values:
                    employee = values[0]
                    logger.info(f"Employee create hit duplicate email; reusing id={employee['id']}")
                    return {"value": employee}
            raise

    if name == "update_employee":
        eid = args["employee_id"]
        fields = args["fields"]
        return await client.put(f"/employee/{eid}", json={"id": eid, **fields})

    if name == "create_customer":
        # Handle flat address fields the model might send
        addr = args.pop("address", None)
        postal_code = args.pop("postalCode", None)
        city = args.pop("city", None)
        if addr or postal_code or city:
            if "postalAddress" not in args:
                args["postalAddress"] = {}
            if addr:
                args["postalAddress"]["addressLine1"] = addr
            if postal_code:
                args["postalAddress"]["postalCode"] = postal_code
            if city:
                args["postalAddress"]["city"] = city
        # Search first by org number — sandbox may have pre-existing customer
        org_number = args.get("organizationNumber")
        if org_number:
            try:
                result = await client.get("/customer", params={
                    "organizationNumber": org_number, "fields": "id,name,organizationNumber,isCustomer,isSupplier"
                })
                values = result.get("values", [])
                if values:
                    customer = values[0]
                    customer_id = customer["id"]
                    update_fields = {}
                    requested_is_customer = args.get("isCustomer")
                    requested_is_supplier = args.get("isSupplier")
                    if requested_is_customer is not None and requested_is_customer != customer.get("isCustomer"):
                        update_fields["isCustomer"] = requested_is_customer
                    if requested_is_supplier is not None and requested_is_supplier != customer.get("isSupplier"):
                        update_fields["isSupplier"] = requested_is_supplier
                    if update_fields:
                        try:
                            logger.info(f"Updating existing customer id={customer_id} for org {org_number}")
                            return await client.put(f"/customer/{customer_id}", json={"id": customer_id, **update_fields})
                        except Exception as update_error:
                            logger.warning(f"Customer update failed for existing customer id={customer_id}; reusing existing entity: {update_error}")
                    logger.info(f"Reusing existing customer id={customer_id} for org {org_number}")
                    return {"value": customer}
            except Exception:
                pass
        # Ensure isCustomer is set unless this is a supplier-only entity
        if "isCustomer" not in args and not args.get("isSupplier"):
            args["isCustomer"] = True
        try:
            return await client.post("/customer", json=args)
        except Exception as e:
            if org_number and _is_duplicate_error(e, "nummeret er i bruk", "already exists"):
                result = await client.get("/customer", params={
                    "organizationNumber": org_number, "fields": "id,name,organizationNumber,isCustomer,isSupplier"
                })
                values = result.get("values", [])
                if values:
                    customer = values[0]
                    logger.info(f"Customer create hit duplicate organization number; reusing id={customer['id']}")
                    return {"value": customer}
            raise

    if name == "update_customer":
        cid = args["customer_id"]
        fields = args["fields"]
        return await client.put(f"/customer/{cid}", json={"id": cid, **fields})

    if name == "create_product":
        product_number = args.get("number")
        # Search first if product number given — avoids 422 error on conflict
        if product_number:
            try:
                result = await client.get("/product", params={"productNumber": product_number, "fields": "id,name,number"})
                values = result.get("values", [])
                if values:
                    product = values[0]
                    product_id = product["id"]
                    logger.info(f"Reusing existing product id={product_id} for number {product_number}")
                    return {"value": product}
            except Exception:
                pass
        try:
            return await client.post("/product", json=args)
        except Exception as e:
            if product_number and _is_duplicate_error(e, "nummeret er i bruk", "already exists"):
                result = await client.get("/product", params={"productNumber": product_number, "fields": "id,name,number"})
                values = result.get("values", [])
                if values:
                    product = values[0]
                    logger.info(f"Product create hit duplicate number; reusing id={product['id']}")
                    return {"value": product}
            raise

    if name == "create_order":
        # Auto-inject customer reference if model omitted it
        if "customer" not in args and ctx and ctx.last_customer_id:
            args["customer"] = {"id": ctx.last_customer_id}
            logger.info(f"Auto-injected customer id={ctx.last_customer_id} into order")
        if "project" not in args and ctx and ctx.last_project_id:
            args["project"] = {"id": ctx.last_project_id}
            logger.info(f"Auto-injected project id={ctx.last_project_id} into order")
        # Auto-inject product references into order lines missing them
        if ctx and ctx.product_ids and "orderLines" in args:
            lines_without_product = [l for l in args["orderLines"] if "product" not in l]
            if len(lines_without_product) == len(ctx.product_ids):
                # Exact match: inject products in order
                for line, pid in zip(lines_without_product, ctx.product_ids):
                    line["product"] = {"id": pid}
                    logger.info(f"Auto-injected product id={pid} into order line")
            elif len(lines_without_product) > 0 and len(ctx.product_ids) == 1:
                # Single product, inject into all lines missing a product
                for line in lines_without_product:
                    line["product"] = {"id": ctx.product_ids[0]}
                    logger.info(f"Auto-injected single product id={ctx.product_ids[0]} into order line")
        # Auto-inject default vatType (25% = id 3) on order lines missing it
        if "orderLines" in args:
            has_ex_vat = False
            has_inc_vat = False
            for line in args["orderLines"]:
                if "vatType" not in line:
                    line["vatType"] = {"id": 3}
                    logger.info("Auto-injected default vatType id=3 (25%) into order line")
                if "unitPriceExcludingVatCurrency" in line:
                    has_ex_vat = True
                if "unitPriceIncludingVatCurrency" in line:
                    has_inc_vat = True
            # Tripletex validates unit price mode against the order-level flag.
            if "isPrioritizeAmountsIncludingVat" not in args:
                if has_inc_vat and not has_ex_vat:
                    args["isPrioritizeAmountsIncludingVat"] = True
                    logger.info("Auto-set isPrioritizeAmountsIncludingVat=true from inclusive order line pricing")
                elif has_ex_vat and not has_inc_vat:
                    args["isPrioritizeAmountsIncludingVat"] = False
                    logger.info("Auto-set isPrioritizeAmountsIncludingVat=false from exclusive order line pricing")
        return await client.post("/order", json=args)

    if name == "create_invoice":
        # Auto-inject orders reference if model omitted it
        if "orders" not in args and ctx and ctx.last_order_id:
            args["orders"] = [{"id": ctx.last_order_id}]
            logger.info(f"Auto-injected order id={ctx.last_order_id} into invoice")
        try:
            return await client.post("/invoice", json=args)
        except Exception as e:
            if "bankkontonummer" in str(e).lower():
                logger.info("Invoice requires bank account — auto-registering")
                await _ensure_bank_account(client)
                return await client.post("/invoice", json=args)
            raise

    if name == "create_project":
        # Auto-inject projectManager if missing
        if "projectManager" not in args and ctx and ctx.last_employee_id:
            args["projectManager"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} as projectManager")
        if "customer" not in args and ctx and ctx.last_customer_id:
            args["customer"] = {"id": ctx.last_customer_id}
            logger.info(f"Auto-injected customer id={ctx.last_customer_id} into project")
        if _is_placeholder_project_number(args.get("number")):
            args["number"] = _generate_project_number()
            logger.info(f"Auto-generated project number {args['number']}")
        if "startDate" not in args:
            args["startDate"] = datetime.date.today().isoformat()
        return await client.post("/project", json=args)

    if name == "create_department":
        return await client.post("/department", json=args)

    if name == "create_travel_expense":
        # Auto-inject employee if missing
        if "employee" not in args and ctx and ctx.last_employee_id:
            args["employee"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} into travel expense")
        # Move date fields into nested travelDetails object
        travel_details = args.pop("travelDetails", {})
        for date_field in ("departureDate", "returnDate", "departureDateTime", "returnDateTime"):
            val = args.pop(date_field, None)
            if val:
                # Normalize field names
                normalized = date_field.replace("DateTime", "Date")
                travel_details[normalized] = val
        if travel_details:
            args["travelDetails"] = travel_details
        return await client.post("/travelExpense", json=args)

    if name == "create_per_diem_compensation":
        if "travelExpense" not in args and ctx and ctx.last_travel_expense_id:
            args["travelExpense"] = {"id": ctx.last_travel_expense_id}
            logger.info(f"Auto-injected travel expense id={ctx.last_travel_expense_id} into per diem compensation")
        if "rateCategory" not in args and ctx and ctx.last_rate_category_id:
            args["rateCategory"] = {"id": ctx.last_rate_category_id}
            logger.info(f"Auto-injected rate category id={ctx.last_rate_category_id} into per diem compensation")
        return await client.post("/travelExpense/perDiemCompensation", json=args)

    if name == "create_travel_cost":
        if "travelExpense" not in args and ctx and ctx.last_travel_expense_id:
            args["travelExpense"] = {"id": ctx.last_travel_expense_id}
            logger.info(f"Auto-injected travel expense id={ctx.last_travel_expense_id} into travel cost")
        if "costCategory" not in args and ctx and ctx.last_cost_category_id:
            args["costCategory"] = {"id": ctx.last_cost_category_id}
            logger.info(f"Auto-injected cost category id={ctx.last_cost_category_id} into travel cost")
        if "paymentType" not in args and ctx and ctx.last_payment_type_id:
            args["paymentType"] = {"id": ctx.last_payment_type_id}
            logger.info(f"Auto-injected payment type id={ctx.last_payment_type_id} into travel cost")
        return await client.post("/travelExpense/cost", json=args)

    if name == "create_project_activity":
        if "project" not in args and ctx and ctx.last_project_id:
            args["project"] = {"id": ctx.last_project_id}
            logger.info(f"Auto-injected project id={ctx.last_project_id} into project activity")
        if "activity" not in args and ctx and ctx.last_activity_id:
            args["activity"] = {"id": ctx.last_activity_id}
            logger.info(f"Auto-injected activity id={ctx.last_activity_id} into project activity")
        return await client.post("/project/projectActivity", json=args)

    if name == "create_timesheet_entry":
        if "employee" not in args and ctx and ctx.last_employee_id:
            args["employee"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} into timesheet entry")
        if "project" not in args and ctx and ctx.last_project_id:
            args["project"] = {"id": ctx.last_project_id}
            logger.info(f"Auto-injected project id={ctx.last_project_id} into timesheet entry")
        if "activity" not in args and ctx and ctx.last_activity_id:
            args["activity"] = {"id": ctx.last_activity_id}
            logger.info(f"Auto-injected activity id={ctx.last_activity_id} into timesheet entry")
        return await client.post("/timesheet/entry", json=args)

    if name == "update_project_hourly_rate":
        hourly_rate_id = args.pop("hourly_rate_id", None)
        if hourly_rate_id is None and ctx and ctx.last_hourly_rate_id:
            hourly_rate_id = ctx.last_hourly_rate_id
            logger.info(f"Auto-injected hourly rate id={ctx.last_hourly_rate_id} into project hourly rate update")
        if hourly_rate_id is None:
            raise ValueError("update_project_hourly_rate requires hourly_rate_id")
        if "project" not in args and ctx and ctx.last_project_id:
            args["project"] = {"id": ctx.last_project_id}
            logger.info(f"Auto-injected project id={ctx.last_project_id} into project hourly rate")
        if "startDate" not in args:
            args["startDate"] = datetime.date.today().isoformat()
        if "hourlyRateModel" not in args:
            args["hourlyRateModel"] = "TYPE_FIXED_HOURLY_RATE"
        if "showInProjectOrder" not in args:
            args["showInProjectOrder"] = True
        return await client.put(f"/project/hourlyRates/{hourly_rate_id}", json={"id": hourly_rate_id, **args})

    if name == "create_accounting_dimension_name":
        return await client.post("/ledger/accountingDimensionName", json=args)

    if name == "create_accounting_dimension_value":
        if "dimensionIndex" not in args and ctx and ctx.last_dimension_index:
            args["dimensionIndex"] = ctx.last_dimension_index
            logger.info(f"Auto-injected dimension index {ctx.last_dimension_index} into accounting dimension value")
        return await client.post("/ledger/accountingDimensionValue", json=args)

    if name == "create_voucher":
        if "postings" in args and isinstance(args["postings"], list):
            for i, posting in enumerate(args["postings"]):
                if isinstance(posting, dict):
                    posting.pop("guiRow", None)
                    posting["row"] = i + 1
        return await client.post("/ledger/voucher", json=args)

    if name == "create_salary_transaction":
        return await client.post("/salary/transaction", json=args)

    if name == "search_entity":
        entity_type = args["entity_type"]
        params = args.get("params", {})
        # GET /invoice requires invoiceDateFrom and invoiceDateTo
        if entity_type == "invoice":
            if "invoiceDateFrom" not in params:
                params["invoiceDateFrom"] = "2000-01-01"
                logger.info("Auto-injected invoiceDateFrom=2000-01-01 for invoice search")
            if "invoiceDateTo" not in params:
                params["invoiceDateTo"] = "2100-01-01"
                logger.info("Auto-injected invoiceDateTo=2100-01-01 for invoice search")
        if not _has_meaningful_search_filters(params):
            logger.warning(f"Blocked unfiltered search_entity call for {entity_type}")
            return {"fullResultSize": 0, "values": []}
        result = await client.get(f"/{entity_type}", params=params)
        # Track first result ID in context for auto-injection
        if ctx:
            values = result.get("values", [])
            if values:
                first_id = values[0].get("id")
                attr_map = {
                    "employee": "last_employee_id",
                    "customer": "last_customer_id",
                    "project": "last_project_id",
                    "invoice": "last_invoice_id",
                }
                attr = attr_map.get(entity_type)
                if attr and first_id:
                    setattr(ctx, attr, first_id)
                    logger.info(f"EntityContext from search: {attr} = {first_id}")
        return result

    if name == "get_entity":
        return await client.get(f"/{args['entity_type']}/{args['entity_id']}")

    if name == "delete_entity":
        return await client.delete(f"/{args['entity_type']}/{args['entity_id']}")

    if name == "tripletex_api_call":
        method = args["method"]
        raw_path = args["path"]
        params = args.get("params") or {}
        body = args.get("body")
        # Block unavailable endpoints to save API calls
        if "/company" in raw_path:
            raise ValueError("/company endpoint is not available in the Tripletex proxy. You do not need company info to complete tasks.")
        # Extract query params embedded in the path (e.g. /invoice/123/:payment?paymentDate=2026-03-20)
        parsed = urlparse(raw_path)
        path = parsed.path
        if parsed.query:
            embedded = parse_qs(parsed.query, keep_blank_values=True)
            for k, v in embedded.items():
                if k not in params:
                    params[k] = v[0]  # parse_qs returns lists; take first value
            logger.info(f"Extracted query params from path: {list(embedded.keys())}")
        if path.startswith("/ledger/vatType") and isinstance(params.get("fields"), str):
            rewritten_fields = _rewrite_fields_filter(params["fields"], {"rate": "percentage"})
            if rewritten_fields != params["fields"]:
                params["fields"] = rewritten_fields
                logger.info(f"Normalized vatType fields filter to {rewritten_fields}")
        if path == "/product" and "number" in params and "productNumber" not in params:
            number_value = params.get("number")
            if isinstance(number_value, str) and not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", number_value):
                params["productNumber"] = number_value
                params.pop("number", None)
                logger.info(f"Normalized product lookup number -> productNumber for value {number_value}")
        # Auto-inject required date range for invoice LIST searches only (not by-ID lookups)
        is_invoice_list = method == "GET" and "/invoice" in path and "/:payment" not in path
        # Skip if path has a numeric ID (e.g. /invoice/2147493584)
        if is_invoice_list and re.search(r'/invoice/\d+', path):
            is_invoice_list = False
        if is_invoice_list:
            if "invoiceDateFrom" not in params and "invoiceDateFrom" not in path:
                params["invoiceDateFrom"] = "2000-01-01"
                logger.info("Auto-injected invoiceDateFrom for invoice search")
            if "invoiceDateTo" not in params and "invoiceDateTo" not in path:
                params["invoiceDateTo"] = "2100-01-01"
                logger.info("Auto-injected invoiceDateTo for invoice search")
        # Fix row numbering in voucher postings — row 0 is reserved by Tripletex
        if body and "postings" in body and isinstance(body["postings"], list):
            for i, posting in enumerate(body["postings"]):
                if isinstance(posting, dict):
                    posting.pop("guiRow", None)
                    posting["row"] = i + 1  # Row must start at 1, not 0
        # Guard: POST/PUT without body causes "Kan ikke være null" errors
        # Exempt action endpoints (/:payment, /:createCreditNote, /:invoice, etc.)
        is_action = "/:" in path
        if method in ("POST", "PUT") and not body and not is_action:
            raise ValueError(
                f"tripletex_api_call {method} {path} requires a 'body' parameter with the JSON payload. "
                f"Example: {{\"method\": \"{method}\", \"path\": \"{path}\", \"body\": {{...your fields here...}}}}"
            )
        async def _run_with_bank_retry(coro_factory):
            """Retry once after registering bank account if needed."""
            try:
                return await coro_factory()
            except Exception as e:
                if "bankkontonummer" in str(e).lower():
                    logger.info("Bank account required — auto-registering via tripletex_api_call path")
                    await _ensure_bank_account(client)
                    return await coro_factory()
                raise

        if method == "GET":
            result = await client.get(path, params=params)
            _track_lookup_context(ctx, path, result)
            return result
        if method == "POST":
            if "/invoice" in path or "/:invoice" in path:
                return await _run_with_bank_retry(lambda: client.post(path, json=body))
            return await client.post(path, json=body)
        if method == "PUT":
            if "/invoice" in path or "/:invoice" in path:
                return await _run_with_bank_retry(lambda: client.put(path, json=body, params=params))
            return await client.put(path, json=body, params=params)
        if method == "DELETE":
            return await client.delete(path)

    if name == "find_tripletex_endpoints":
        if endpoint_search is None:
            raise RuntimeError("Azure AI Search is not configured for endpoint discovery")
        return await endpoint_search.search_endpoints(
            task=args["task"],
            method=args.get("method"),
            top_k=args.get("top_k"),
        )

    raise ValueError(f"Unknown tool: {name}")
