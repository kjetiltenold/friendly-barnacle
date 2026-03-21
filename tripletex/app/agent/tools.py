"""Tool definitions for OpenAI tool-use and dispatch to Tripletex API."""

import datetime
import json
import logging
import re
import uuid
import unicodedata
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
    last_sales_customer_id: int | None = None
    last_supplier_id: int | None = None
    last_product_id: int | None = None
    product_ids: list[int] | None = None  # All product IDs created/found
    project_ids: list[int] | None = None
    activity_ids: list[int] | None = None
    employee_ids: list[int] | None = None
    employee_snapshots: dict[int, dict] | None = None
    last_order_id: int | None = None
    last_employee_id: int | None = None
    last_employment_id: int | None = None
    last_employment_details_id: int | None = None
    last_project_id: int | None = None
    last_invoice_id: int | None = None
    last_travel_expense_id: int | None = None
    last_travel_expense_departure_date: str | None = None
    last_travel_expense_return_date: str | None = None
    last_activity_id: int | None = None
    last_rate_category_id: int | None = None
    last_cost_category_id: int | None = None
    last_cost_categories: list[dict] | None = None
    last_payment_type_id: int | None = None
    last_hourly_rate_id: int | None = None
    last_dimension_index: int | None = None
    last_dimension_value_id: int | None = None
    last_department_id: int | None = None
    last_standard_time_id: int | None = None
    last_voucher_id: int | None = None
    last_vat_type_id: int | None = None
    last_account_id: int | None = None
    account_cache: dict[int, dict] | None = None
    vat_type_cache: dict | None = None  # Maps (percentage, direction_hint) -> vat_type_id
    project_start_dates: dict[int, str] | None = None
    linked_project_activity_pairs: set[tuple[int, int]] | None = None
    timesheet_hours_by_day: dict[tuple[int, int, int, str], float] | None = None
    last_top_expense_analysis_key: str | None = None
    last_top_expense_analysis: dict | None = None
    next_project_activity_pair_index: int = 0
    travel_cost_count: int = 0
    prompt_text: str | None = None

    def __post_init__(self):
        if self.product_ids is None:
            self.product_ids = []
        if self.project_ids is None:
            self.project_ids = []
        if self.activity_ids is None:
            self.activity_ids = []
        if self.employee_ids is None:
            self.employee_ids = []
        if self.employee_snapshots is None:
            self.employee_snapshots = {}
        if self.account_cache is None:
            self.account_cache = {}
        if self.last_cost_categories is None:
            self.last_cost_categories = []
        if self.project_start_dates is None:
            self.project_start_dates = {}
        if self.linked_project_activity_pairs is None:
            self.linked_project_activity_pairs = set()
        if self.timesheet_hours_by_day is None:
            self.timesheet_hours_by_day = {}

    def track(self, name: str, result: dict, request_args: dict | None = None) -> None:
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
            "create_employment_details": "last_employment_details_id",
            "create_project": "last_project_id",
            "create_activity": "last_activity_id",
            "create_invoice": "last_invoice_id",
            "create_travel_expense": "last_travel_expense_id",
            "create_department": "last_department_id",
            "create_standard_time": "last_standard_time_id",
            "create_voucher": "last_voucher_id",
        }
        attr = mapping.get(name)
        if attr:
            setattr(self, attr, entity_id)
            logger.info(f"EntityContext: {attr} = {entity_id}")
        if name == "create_customer":
            requested_is_customer = None
            requested_is_supplier = None
            if isinstance(request_args, dict):
                requested_is_customer = request_args.get("isCustomer")
                requested_is_supplier = request_args.get("isSupplier")
            if requested_is_customer is True or (
                requested_is_customer is None and value.get("isCustomer") is not False
            ):
                self.last_sales_customer_id = entity_id
                logger.info(f"EntityContext: last_sales_customer_id = {entity_id}")
            if requested_is_supplier is True or value.get("isSupplier") is True:
                self.last_supplier_id = entity_id
                logger.info(f"EntityContext: last_supplier_id = {entity_id}")
        if name == "create_employee":
            if entity_id not in self.employee_ids:
                self.employee_ids.append(entity_id)
            self.employee_snapshots[entity_id] = {
                "id": entity_id,
                "firstName": value.get("firstName"),
                "lastName": value.get("lastName"),
                "email": value.get("email"),
            }
            employments = value.get("employments") or []
            if employments and isinstance(employments[0], dict):
                employment_id = employments[0].get("id")
                if employment_id is not None:
                    self.last_employment_id = employment_id
                    logger.info(f"EntityContext: last_employment_id = {employment_id}")
        if name == "create_employment_details":
            employment = value.get("employment") or {}
            employment_id = employment.get("id")
            if employment_id is not None:
                self.last_employment_id = employment_id
        if name == "create_standard_time":
            employee = value.get("employee") or {}
            employee_id = employee.get("id")
            if employee_id is not None:
                self.last_employee_id = employee_id
        if name == "create_project_activity":
            activity = value.get("activity") or {}
            activity_id = activity.get("id")
            if activity_id is not None:
                self.last_activity_id = activity_id
                logger.info(f"EntityContext: last_activity_id = {activity_id}")
            project = value.get("project") or {}
            project_id = project.get("id")
            if project_id is not None:
                self.last_project_id = project_id
                logger.info(f"EntityContext: last_project_id = {project_id}")
            if project_id is not None and activity_id is not None:
                self.linked_project_activity_pairs.add((project_id, activity_id))
        # Track all product IDs for multi-product orders
        if name == "create_product" and entity_id not in self.product_ids:
            self.product_ids.append(entity_id)
        if name == "create_project" and entity_id not in self.project_ids:
            self.project_ids.append(entity_id)
            start_date = value.get("startDate")
            if isinstance(start_date, str):
                self.project_start_dates[entity_id] = start_date
        if name == "create_activity" and entity_id not in self.activity_ids:
            self.activity_ids.append(entity_id)
        if name == "create_timesheet_entry":
            employee_id = _extract_reference_id(value.get("employee"))
            project_id = _extract_reference_id(value.get("project"))
            activity_id = _extract_reference_id(value.get("activity"))
            entry_date = value.get("date")
            hours = _coerce_number(value.get("hours"))
            if (
                employee_id is not None
                and project_id is not None
                and activity_id is not None
                and isinstance(entry_date, str)
            ):
                key = (employee_id, project_id, activity_id, entry_date)
                self.timesheet_hours_by_day[key] = round(
                    self.timesheet_hours_by_day.get(key, 0.0) + hours,
                    2,
                )
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
    first = values[0] if values else result.get("value")
    if not isinstance(first, dict):
        return
    first_id = first.get("id")
    if first_id is None:
        return

    if path.startswith("/activity"):
        ctx.last_activity_id = first_id
    elif path.startswith("/travelExpense/rateCategory"):
        ctx.last_rate_category_id = first_id
    elif path.startswith("/travelExpense/costCategory"):
        ctx.last_cost_category_id = first_id
        ctx.last_cost_categories = [item for item in values if isinstance(item, dict)]
    elif (
        path.startswith("/travelExpense/paymentType")
        or path.startswith("/invoice/paymentType")
        or path.startswith("/ledger/paymentTypeOut")
    ):
        ctx.last_payment_type_id = first_id
    elif path.startswith("/project/hourlyRates"):
        ctx.last_hourly_rate_id = first_id
    elif path.startswith("/employee/employment/details"):
        ctx.last_employment_details_id = first_id
        employment = first.get("employment") or {}
        employment_id = employment.get("id")
        if employment_id is not None:
            ctx.last_employment_id = employment_id
        employee = employment.get("employee") or {}
        employee_id = employee.get("id")
        if employee_id is not None:
            ctx.last_employee_id = employee_id
    elif path == "/employee/employment" or re.fullmatch(r"/employee/employment/\d+", path):
        ctx.last_employment_id = first_id
        employee = first.get("employee") or {}
        employee_id = employee.get("id")
        if employee_id is not None:
            ctx.last_employee_id = employee_id
    elif path.startswith("/employee/standardTime"):
        ctx.last_standard_time_id = first_id
        employee = first.get("employee") or {}
        employee_id = employee.get("id")
        if employee_id is not None:
            ctx.last_employee_id = employee_id
    elif path == "/invoice" or re.fullmatch(r"/invoice/\d+", path):
        ctx.last_invoice_id = first_id
        customer = first.get("customer") or {}
        customer_id = customer.get("id")
        if customer_id is not None:
            ctx.last_customer_id = customer_id
            ctx.last_sales_customer_id = customer_id
    elif path == "/project" or re.fullmatch(r"/project/\d+", path):
        ctx.last_project_id = first_id
        start_date = first.get("startDate")
        if isinstance(start_date, str):
            ctx.project_start_dates[first_id] = start_date
    elif path.startswith("/department"):
        ctx.last_department_id = first_id
    elif path.startswith("/ledger/vatType"):
        ctx.last_vat_type_id = first_id
        # Cache all VAT types from the response for auto-selection
        if values and ctx.vat_type_cache is None:
            ctx.vat_type_cache = {}
        if values:
            for vt in values:
                vt_id = vt.get("id")
                pct = vt.get("percentage")
                name_lower = str(vt.get("name", "")).lower()
                if vt_id is not None and pct is not None:
                    direction = "incoming" if "inng" in name_lower or "incoming" in name_lower else "outgoing"
                    ctx.vat_type_cache[(pct, direction)] = vt_id
    elif path.startswith("/ledger/account"):
        ctx.last_account_id = first_id
        accounts = values if values else [first]
        for account in accounts:
            if not isinstance(account, dict):
                continue
            account_id = account.get("id")
            if account_id is not None:
                ctx.account_cache[account_id] = account


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
            "nationalIdentityNumber": {"type": "string", "description": "National identity number / personnummer"},
            "dnumber": {"type": "string", "description": "D-number if applicable"},
            "department": {"type": "object", "description": "{\"id\": department_id}"},
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
            "vatPercentage": {"type": "number", "description": "Convenience alias. The executor resolves this outgoing VAT percentage to vatType automatically."},
            "vatRate": {"type": "number", "description": "Alias for vatPercentage."},
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
            "isInternal": {"type": "boolean"},
            "isFixedPrice": {"type": "boolean"},
            "fixedprice": {"type": "number", "description": "Fixed project price amount in project currency."},
            "fixedPrice": {"type": "number", "description": "Convenience alias for fixedprice."},
            "isClosed": {"type": "boolean"},
        },
        "required": ["name"],
    }),
    _tool("create_activity", "Create an activity in Tripletex. Reuses an existing activity with the same name when found.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "isInactive": {"type": "boolean"},
        },
        "required": ["name"],
    }),
    _tool("create_department", "Create a department in Tripletex.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "departmentNumber": {"type": "string"},
        },
        "required": ["name"],
    }),
    _tool("create_employment_details", "Create or update employment details for an employment. Use this for annual salary, employment percentage, and working-hours scheme. If hoursPerDay is included, the executor will also update employee/standardTime.", {
        "type": "object",
        "properties": {
            "employment": {"type": "object", "description": "{\"id\": employment_id}"},
            "employmentId": {"type": "integer"},
            "employee": {"type": "object", "description": "Optional employee reference used for department and standard-time updates."},
            "employeeId": {"type": "integer"},
            "date": {"type": "string", "description": "Effective date YYYY-MM-DD"},
            "fromDate": {"type": "string", "description": "Alias for date"},
            "annualSalary": {"type": "number"},
            "salary": {"type": "number", "description": "Alias for annualSalary"},
            "percentageOfFullTimeEquivalent": {"type": "number"},
            "employmentPercentage": {"type": "number", "description": "Alias for percentageOfFullTimeEquivalent"},
            "remunerationType": {"type": "string", "enum": ["MONTHLY_WAGE", "HOURLY_WAGE", "COMMISION_PERCENTAGE", "FEE", "NOT_CHOSEN", "PIECEWORK_WAGE"]},
            "employmentType": {"type": "string", "enum": ["ORDINARY", "MARITIME", "FREELANCE", "NOT_CHOSEN"]},
            "employmentForm": {"type": "string", "enum": ["PERMANENT", "TEMPORARY", "PERMANENT_AND_HIRED_OUT", "TEMPORARY_AND_HIRED_OUT", "TEMPORARY_ON_CALL", "NOT_CHOSEN"]},
            "workingHoursScheme": {"type": "string", "enum": ["NOT_SHIFT", "ROUND_THE_CLOCK", "SHIFT_365", "OFFSHORE_336", "CONTINUOUS", "OTHER_SHIFT", "NOT_CHOSEN"]},
            "workingHoursSchemeId": {"type": "integer", "description": "Optional ID from GET /employee/employment/workingHoursScheme; executor resolves it to the enum value."},
            "hoursPerDay": {"type": "number", "description": "Convenience alias. Applied through employee/standardTime, not employment/details."},
            "hoursPerWeek": {"type": "number", "description": "Convenience alias. Converted to hoursPerDay by dividing by 5."},
            "department": {"type": "object", "description": "Convenience alias. Applied to employee.department, not employment/details."},
            "departmentId": {"type": "integer", "description": "Convenience alias for department.id"},
            "hourlyWage": {"type": "number"},
            "shiftDurationHours": {"type": "number"},
            "occupationCode": {"type": "object", "description": "{\"id\": occupation_code_id}"},
            "occupationCodeCode": {"type": "string", "description": "STYRK/occupation code from the contract. The executor resolves it to occupationCode.id."},
            "occupationCodeName": {"type": "string", "description": "Occupation title/name from the contract. The executor resolves it to occupationCode.id."},
            "stillingskode": {"type": "string", "description": "Alias for occupationCodeCode or occupationCodeName from Norwegian contracts."},
            "payrollTaxMunicipalityId": {"type": "object", "description": "{\"id\": municipality_id}"},
        },
    }),
    _tool("create_standard_time", "Create or update employee standard working time. Use this for hoursPerDay.", {
        "type": "object",
        "properties": {
            "employee": {"type": "object", "description": "{\"id\": employee_id}"},
            "employeeId": {"type": "integer"},
            "fromDate": {"type": "string", "description": "Effective date YYYY-MM-DD"},
            "date": {"type": "string", "description": "Alias for fromDate"},
            "startDate": {"type": "string", "description": "Alias for fromDate"},
            "hoursPerDay": {"type": "number"},
            "hoursPerWeek": {"type": "number", "description": "Converted to hoursPerDay by dividing by 5."},
        },
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
            "budgetHours": {"type": "number"},
            "budgetHourlyRateCurrency": {"type": "number"},
            "budgetFeeCurrency": {"type": "number"},
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
                        "department": {"type": "object", "description": "{\"id\": department_id}"},
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

BASE_TOOL_DEFINITIONS.extend([
    _tool("find_top_expense_account_increases", "Analyze ledger postings across two periods and return the expense accounts with the largest increase from period A to period B. This tool is read-only. If the task also asks you to create projects, activities, or other entities from the result, you must continue with those write tools after this analysis.", {
        "type": "object",
        "properties": {
            "period_a_from": {"type": "string", "description": "YYYY-MM-DD inclusive"},
            "period_a_to": {"type": "string", "description": "YYYY-MM-DD exclusive"},
            "period_b_from": {"type": "string", "description": "YYYY-MM-DD inclusive"},
            "period_b_to": {"type": "string", "description": "YYYY-MM-DD exclusive"},
            "top_n": {"type": "integer", "default": 3},
        },
        "required": ["period_a_from", "period_a_to", "period_b_from", "period_b_to"],
    }),
    _tool("delete_travel_expense", "Delete a travel expense report. Searches by employee email if travel_expense_id is not given. If multiple reports exist, provide title to disambiguate.", {
        "type": "object",
        "properties": {
            "travel_expense_id": {"type": "integer", "description": "ID of the travel expense to delete. If omitted, the tool searches by employee_email."},
            "employee_email": {"type": "string", "description": "Employee email to find travel expenses for deletion."},
            "title": {"type": "string", "description": "Optional travel expense title to choose the correct report when an employee has multiple reports."},
        },
    }),
    _tool("reverse_voucher", "Reverse a voucher in the ledger. Creates a reversal entry on the given date.", {
        "type": "object",
        "properties": {
            "voucher_id": {"type": "integer", "description": "ID of the voucher to reverse."},
            "date": {"type": "string", "description": "Reversal date YYYY-MM-DD"},
        },
        "required": ["voucher_id", "date"],
    }),
])

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
    for raw_part in _split_filter_parts(fields):
        part = raw_part.strip()
        if not part:
            continue
        if "." in part and "(" not in part and part.count(".") == 1:
            parent, child = part.split(".", 1)
            part = f"{parent}({child})"
        normalized = replacements.get(part, part)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        rewritten.append(normalized)
    return ",".join(rewritten)


def _rewrite_sorting_filter(sorting: str, replacements: dict[str, str]) -> str:
    """Rewrite invalid sorting aliases while preserving order and sign."""
    rewritten: list[str] = []
    seen: set[str] = set()
    for raw_part in sorting.split(","):
        part = raw_part.strip()
        if not part:
            continue
        prefix = ""
        key = part
        if part[0] in "+-":
            prefix = part[0]
            key = part[1:]
        normalized = replacements.get(key, key)
        if not normalized:
            continue
        rewritten_part = f"{prefix}{normalized}"
        if rewritten_part in seen:
            continue
        seen.add(rewritten_part)
        rewritten.append(rewritten_part)
    return ",".join(rewritten)


def _extend_fields_filter(fields: str, extra_fields: list[str]) -> str:
    """Append extra fields to a fields filter if they are missing."""
    base_fields = _rewrite_fields_filter(fields, {})
    current = [field for field in _split_filter_parts(base_fields) if field]
    seen = set(current)
    for field in extra_fields:
        if field not in seen:
            current.append(field)
            seen.add(field)
    return ",".join(current)


def _split_filter_parts(value: str) -> list[str]:
    """Split a Tripletex fields/sorting filter on top-level commas only."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_ledger_voucher_fields_filter(fields: str) -> str:
    """Normalize common invalid ledger/voucher fields to schema-valid Voucher/Posting fields."""
    top_level_replacements = {
        "voucherNumber": "number",
        "voucherTempNumber": "tempNumber",
        "voucherYear": "year",
    }
    posting_replacements = {
        "voucherNumber": "voucher(number)",
        "voucherTempNumber": "voucher(tempNumber)",
        "voucherYear": "voucher(year)",
    }
    rewritten: list[str] = []
    seen: set[str] = set()
    for raw_part in _split_filter_parts(fields):
        part = raw_part.strip()
        if not part:
            continue
        if part.startswith("postings(") and part.endswith(")"):
            inner = part[len("postings("):-1]
            rewritten_inner: list[str] = []
            inner_seen: set[str] = set()
            for raw_child in _split_filter_parts(inner):
                child = raw_child.strip()
                if not child:
                    continue
                normalized_child = posting_replacements.get(child, child)
                if normalized_child in inner_seen:
                    continue
                inner_seen.add(normalized_child)
                rewritten_inner.append(normalized_child)
            part = f"postings({','.join(rewritten_inner)})"
        else:
            part = top_level_replacements.get(part, part)
        if part in seen:
            continue
        seen.add(part)
        rewritten.append(part)
    return ",".join(rewritten)


def _normalize_exclusive_date_range(path: str, params: dict) -> None:
    """Adjust common inclusive end-date mistakes for exclusive-range Tripletex endpoints."""
    if path != "/ledger/posting":
        return
    date_from_raw = params.get("dateFrom")
    date_to_raw = params.get("dateTo")
    if not isinstance(date_from_raw, str) or not isinstance(date_to_raw, str):
        return
    try:
        date_from = datetime.date.fromisoformat(date_from_raw)
        date_to = datetime.date.fromisoformat(date_to_raw)
    except ValueError:
        return
    if date_from.day != 1:
        return
    next_month = (date_from.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    month_end = next_month - datetime.timedelta(days=1)
    if date_to == month_end:
        params["dateTo"] = next_month.isoformat()
        logger.info(
            f"Normalized {path} dateTo from inclusive month-end {date_to_raw} to exclusive {params['dateTo']}"
        )


def _find_cached_account_by_number(ctx: EntityContext | None, number: str) -> dict | None:
    if ctx is None or not ctx.account_cache:
        return None
    target = str(number)
    for account in ctx.account_cache.values():
        if str(account.get("number")) == target:
            return account
    return None


def _normalize_year_end_depreciation_postings(postings: list[dict] | None, description: str | None, ctx: EntityContext | None) -> None:
    if ctx is None or not isinstance(postings, list):
        return
    normalized_description = _normalize_prompt_text(description)
    if not any(token in normalized_description for token in ("avskriv", "depreciat", "abschreib", "amortis")):
        return
    if len(postings) != 2:
        return
    debit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    credit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(debit_postings) != 1 or len(credit_postings) != 1:
        return
    preferred_expense = _find_cached_account_by_number(ctx, "6010")
    preferred_accumulated = _find_cached_account_by_number(ctx, "1209")
    debit_posting = debit_postings[0]
    credit_posting = credit_postings[0]
    debit_account = _get_cached_account(ctx, debit_posting) or {}
    credit_account = _get_cached_account(ctx, credit_posting) or {}
    if preferred_expense and str(debit_account.get("number")) != "6010":
        debit_posting["account"] = {"id": preferred_expense["id"]}
        logger.info("Normalized depreciation expense posting to account 6010 from cached lookup")
    credit_number = str(credit_account.get("number") or "")
    if preferred_accumulated and credit_number != "1209" and credit_number.startswith("12"):
        credit_posting["account"] = {"id": preferred_accumulated["id"]}
        logger.info("Normalized accumulated depreciation posting to account 1209 from cached lookup")


def _classify_month_end_closing_pair(postings: list[dict] | None, description: str | None) -> str | None:
    pair_text = " ".join(
        str(posting.get("description") or "")
        for posting in (postings or [])
        if isinstance(posting, dict)
    ).lower()
    fallback_text = str(description or "").lower()
    for text in (pair_text, fallback_text):
        if any(token in text for token in ("salary accrual", "accrued salary", "gehaltsruckstellung", "gehaltsrueckstellung", "gehaltsrückstellung")):
            return "salary accrual"
        if any(token in text for token in ("depreciat", "avskriv", "abschreib", "amortis")):
            return "depreciation"
        if any(token in text for token in ("accrual reversal", "prepaid", "forskudds", "rechnungsabgrenz")):
            return "accrual reversal"
    return None


def _extract_prompt_month_end_accrual_amount(ctx: EntityContext | None) -> float | None:
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    normalized_text = unicodedata.normalize("NFKD", raw_text.lower())
    normalized_text = "".join(ch for ch in normalized_text if not unicodedata.combining(ch))
    match = re.search(
        r"([0-9][0-9\s\u00a0.,']*)\s*(?:nok|kr|eur)?\D{0,20}(?:per\s+month|pro\s+monat|par\s+mois|por\s+mes|per\s+maned)",
        normalized_text,
    )
    if not match:
        return None
    token = match.group(1).strip()
    if not token:
        return None
    token = token.replace("\u00a0", "").replace(" ", "").replace("'", "")
    if token.count(",") and token.count("."):
        last_comma = token.rfind(",")
        last_dot = token.rfind(".")
        decimal_separator = "," if last_comma > last_dot else "."
        thousands_separator = "." if decimal_separator == "," else ","
        token = token.replace(thousands_separator, "")
        if decimal_separator == ",":
            token = token.replace(",", ".")
    elif token.count(",") == 1 and token.count(".") == 0:
        left, right = token.split(",", 1)
        token = f"{left}.{right}" if len(right) <= 2 else f"{left}{right}"
    elif token.count(".") > 1:
        token = token.replace(".", "")
    elif token.count(".") == 1:
        left, right = token.split(".", 1)
        if len(right) > 2:
            token = f"{left}{right}"
    amount = _coerce_number(token)
    return amount if amount > 0 else None


def _preferred_month_end_expense_account(ctx: EntityContext | None) -> dict | None:
    preferred = _find_cached_account_by_number(ctx, "6000")
    if preferred is not None:
        return preferred
    if ctx is None or not ctx.account_cache:
        return None
    candidates: list[tuple[int, dict]] = []
    for account in ctx.account_cache.values():
        number_raw = str(account.get("number") or "")
        if not number_raw.isdigit():
            continue
        number = int(number_raw)
        if 6000 <= number < 8000 and number not in {6030}:
            candidates.append((number, account))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _extract_prompt_year_end_prepaid_amount(ctx: EntityContext | None, account_number: str = "1700") -> float | None:
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    def _parse_prompt_amount(value: str) -> float | None:
        text = str(value or "").strip().replace("\u00a0", " ")
        if not text:
            return None
        text = re.sub(r"\s+", "", text)
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            if text.count(",") == 1 and len(text.split(",")[-1]) in (1, 2):
                text = text.replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "." in text and text.count(".") > 1:
            text = text.replace(".", "")
        try:
            return float(text)
        except ValueError:
            return None
    amount_pattern = r"(\d[\d\s.,]*)"
    account_pattern = re.escape(str(account_number))
    total_pattern = r"(?:total|totalt|totale?n?|insgesamt|summe|summa)"
    location_pattern = r"(?:na|no|on|i|in|au|sur|pa|på|auf)"
    account_word_pattern = r"(?:conta|konto|account|compte)"
    patterns = (
        rf"{total_pattern}\s+{amount_pattern}\s*(?:nok|kr)\b[^.\n)]{{0,80}}?(?:{location_pattern})\s+(?:{account_word_pattern})\s+{account_pattern}\b",
        rf"{account_pattern}\b[^.\n)]{{0,80}}?\b{total_pattern}\s+{amount_pattern}\s*(?:nok|kr)\b",
        rf"{account_pattern}\b[^.\n)]{{0,80}}?{amount_pattern}\s*(?:nok|kr)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if not match:
            continue
        amount = _parse_prompt_amount(match.group(1))
        if amount is not None:
            return round(amount, 2)
    return None


def _extract_prompt_tax_provision_accounts(ctx: EntityContext | None) -> tuple[str, str] | None:
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    text = str(raw_text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    match = re.search(
        r"(?:tax\s+provision|skattekostnad|skatteavsetning|steuerruckstellung|steuerrueckstellung|"
        r"steuer(?:ruckstellung|rueckstellung)|provision\s+d[' ]impot|provision\s+fiscale|provisao\s+fiscal)"
        r"[^.\n]{0,200}?(?:konto|conta|compte|account)\s*(\d{4})\s*/\s*(\d{4})",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1), match.group(2)


def _normalize_year_end_prepaid_reversal_postings(
    postings: list[dict] | None,
    description: str | None,
    ctx: EntityContext | None,
) -> None:
    if ctx is None or not isinstance(postings, list) or len(postings) != 2:
        return
    normalized_description = _normalize_prompt_text(description)
    if not any(
        token in normalized_description
        for token in (
            "prepaid",
            "forskudds",
            "vorausbezahlt",
            "auflos",
            "tilbakeforing",
            "reversal",
            "extourne",
            "cca",
            "chargeconstateedavance",
            "chargesconstateesdavance",
        )
    ):
        return
    target_amount = _extract_prompt_year_end_prepaid_amount(ctx, "1700")
    if target_amount in (None, 0):
        return
    debit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    credit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(debit_postings) != 1 or len(credit_postings) != 1:
        return
    debit_posting = debit_postings[0]
    credit_posting = credit_postings[0]
    debit_account = _get_cached_account(ctx, debit_posting) or {}
    credit_account = _get_cached_account(ctx, credit_posting) or {}
    if str(debit_account.get("number") or "") != "1700" and str(credit_account.get("number") or "") != "1700":
        return
    current_amount = abs(_coerce_number(debit_posting.get("amountGross")))
    if abs(current_amount - target_amount) <= 0.01:
        return
    debit_posting["amountGross"] = target_amount
    debit_posting["amountGrossCurrency"] = target_amount
    credit_posting["amountGross"] = -target_amount
    credit_posting["amountGrossCurrency"] = -target_amount
    logger.info(
        "Normalized year-end prepaid reversal on 1700 to prompt total %.2f from current %.2f",
        target_amount,
        current_amount,
    )


def _normalize_year_end_tax_provision_postings(
    postings: list[dict] | None,
    description: str | None,
    ctx: EntityContext | None,
) -> None:
    if ctx is None or not isinstance(postings, list) or len(postings) != 2:
        return
    prompt_accounts = _extract_prompt_tax_provision_accounts(ctx)
    if prompt_accounts is None:
        return
    normalized_description = _normalize_prompt_text(description)
    if not any(
        token in normalized_description
        for token in (
            "taxprovision",
            "skattekostnad",
            "skatteavsetning",
            "steuerruckstellung",
            "steuerrueckstellung",
            "provisiondimpot",
            "provisionfiscale",
            "provisaofiscal",
        )
    ):
        return
    debit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    credit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(debit_postings) != 1 or len(credit_postings) != 1:
        return
    debit_number, credit_number = prompt_accounts
    preferred_debit = _find_cached_account_by_number(ctx, debit_number)
    preferred_credit = _find_cached_account_by_number(ctx, credit_number)
    if preferred_debit is None or preferred_credit is None:
        return
    debit_posting = debit_postings[0]
    credit_posting = credit_postings[0]
    current_debit_account = _get_cached_account(ctx, debit_posting) or {}
    current_credit_account = _get_cached_account(ctx, credit_posting) or {}
    changed = False
    if str(current_debit_account.get("number") or "") != debit_number:
        debit_posting["account"] = {"id": preferred_debit["id"]}
        changed = True
    if str(current_credit_account.get("number") or "") != credit_number:
        credit_posting["account"] = {"id": preferred_credit["id"]}
        changed = True
    if changed:
        logger.info(
            "Normalized year-end tax provision postings to accounts %s/%s from prompt",
            debit_number,
            credit_number,
        )


def _validate_ledger_error_correction_postings(
    postings: list[dict] | None,
    description: str | None,
    ctx: EntityContext | None,
) -> dict | None:
    if ctx is None or not isinstance(postings, list) or len(postings) < 2:
        return None
    normalized_prompt = _normalize_prompt_text(ctx.prompt_text)
    if not any(
        token in normalized_prompt
        for token in (
            "generalledger",
            "ledgererrors",
            "findthe4errors",
            "grandlivre",
            "trouvezles4erreurs",
            "hauptbuch",
            "findedie4fehler",
            "livromaior",
            "livrorazao",
            "encontreos4erros",
            "libromayor",
            "encuentralos4errores",
            "hovedbok",
            "finnende4feilene",
        )
    ):
        return None
    normalized_description = _normalize_prompt_text(description)
    account_numbers = {
        str((_get_cached_account(ctx, posting) or {}).get("number") or "")
        for posting in postings
        if isinstance(posting, dict)
    }
    suspicious_numbers = sorted(
        number for number in account_numbers if number in {"1920", "2400", "2050", "2990"}
    )
    if not suspicious_numbers:
        return None
    if any(token in normalized_description for token in ("duplicatevoucher", "pieceendouble", "duplicertbilag", "doppelterbeleg", "lançamentoduplicado", "asientoduplicado")):
        logger.warning(
            "Blocked duplicate-voucher correction using guessed balancing account(s): %s",
            ", ".join(suspicious_numbers),
        )
        return {
            "error": (
                "Duplicate-voucher corrections must identify the duplicate voucher and use reverse_voucher "
                "on that voucher ID after reviewing the original voucher/postings. Do not create a manual "
                "balancing line on guessed bank/liability accounts such as 1920, 2400, 2050, or 2990."
            )
        }
    if any(
        token in normalized_description
        for token in (
            "overstatedamount",
            "incorrectamount",
            "wrongamount",
            "montantincorrect",
            "mauvaismontant",
            "feilbelop",
            "falscherbetrag",
            "valorincorreto",
            "importeincorrecto",
        )
    ):
        logger.warning(
            "Blocked wrong-amount correction using guessed balancing account(s): %s",
            ", ".join(suspicious_numbers),
        )
        return {
            "error": (
                "Wrong-amount corrections must use the original voucher's counterpart account and correct only "
                "the delta. Review the original voucher/postings first; do not guess bank/liability accounts "
                "such as 1920, 2400, 2050, or 2990."
            )
        }
    return None


def _normalize_month_end_closing_pair_postings(
    postings: list[dict] | None,
    description: str | None,
    ctx: EntityContext | None,
) -> None:
    if ctx is None or not isinstance(postings, list) or len(postings) != 2:
        return
    topic = _classify_month_end_closing_pair(postings, description)
    if topic is None:
        return
    debit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    credit_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(debit_postings) != 1 or len(credit_postings) != 1:
        return
    debit_posting = debit_postings[0]
    credit_posting = credit_postings[0]
    debit_account = _get_cached_account(ctx, debit_posting) or {}
    credit_account = _get_cached_account(ctx, credit_posting) or {}
    if topic == "depreciation":
        preferred_expense = _find_cached_account_by_number(ctx, "6030") or _find_cached_account_by_number(ctx, "6010")
        if preferred_expense and str(debit_account.get("number")) != str(preferred_expense.get("number")):
            debit_posting["account"] = {"id": preferred_expense["id"]}
            logger.info(
                f"Normalized month-end depreciation expense posting to account {preferred_expense['number']} from cached lookup"
            )
        preferred_accumulated = _find_cached_account_by_number(ctx, "1209")
        if preferred_accumulated and str(credit_account.get("number")) != "1209":
            credit_posting["account"] = {"id": preferred_accumulated["id"]}
            logger.info("Normalized month-end depreciation credit posting to account 1209 from cached lookup")
        return
    if topic == "salary accrual":
        preferred_expense = _find_cached_account_by_number(ctx, "5000")
        preferred_accrued = _find_cached_account_by_number(ctx, "2900")
        if preferred_expense and str(debit_account.get("number")) != "5000":
            debit_posting["account"] = {"id": preferred_expense["id"]}
            logger.info("Normalized salary accrual debit posting to account 5000 from cached lookup")
        if preferred_accrued and str(credit_account.get("number")) != "2900":
            credit_posting["account"] = {"id": preferred_accrued["id"]}
            logger.info("Normalized salary accrual credit posting to account 2900 from cached lookup")
        return
    preferred_prepaid = None
    text = " ".join(
        [str(description or "")]
        + [
            str(posting.get("description") or "")
            for posting in postings
            if isinstance(posting, dict)
        ]
    ).lower()
    prompt_text = str((ctx.prompt_text if ctx else None) or "").lower()
    combined_text = f"{text} {prompt_text}".strip()
    if "1700" in combined_text:
        preferred_prepaid = _find_cached_account_by_number(ctx, "1700")
    elif "1720" in combined_text:
        preferred_prepaid = _find_cached_account_by_number(ctx, "1720")
    if preferred_prepaid and str(credit_account.get("number")) != str(preferred_prepaid.get("number")):
        credit_posting["account"] = {"id": preferred_prepaid["id"]}
        logger.info(
            "Normalized accrual-reversal credit posting to account %s from cached lookup",
            preferred_prepaid["number"],
        )
    prompt_amount = _extract_prompt_month_end_accrual_amount(ctx)
    preferred_expense = _preferred_month_end_expense_account(ctx)
    debit_number = str(debit_account.get("number") or "")
    if (
        prompt_amount is not None
        and preferred_expense is not None
        and debit_number == str(int(round(prompt_amount)))
        and debit_number != str(preferred_expense.get("number"))
    ):
        debit_posting["account"] = {"id": preferred_expense["id"]}
        logger.info(
            "Normalized accrual-reversal debit posting from amount-like account %s to expense account %s from cached lookup",
            debit_number,
            preferred_expense["number"],
        )


def _split_month_end_closing_vouchers(args: dict) -> list[dict] | None:
    postings = args.get("postings")
    description = str(args.get("description") or "")
    normalized_description = description.lower()
    if not isinstance(postings, list) or len(postings) < 4 or len(postings) % 2 != 0:
        return None
    if "month-end" not in normalized_description or "closing" not in normalized_description:
        return None
    base_description = description.split(" - ", 1)[0].strip() if " - " in description else description
    vouchers: list[dict] = []
    for index in range(0, len(postings), 2):
        pair = postings[index:index + 2]
        if len(pair) != 2 or not all(isinstance(posting, dict) for posting in pair):
            return None
        if abs(sum(_coerce_number(posting.get("amountGross")) for posting in pair)) > 0.01:
            return None
        topic = _classify_month_end_closing_pair(pair, description)
        if topic is None:
            return None
        voucher_args = {
            key: json.loads(json.dumps(value))
            for key, value in args.items()
            if key != "postings"
        }
        voucher_args["description"] = f"{base_description} - {topic}"
        voucher_args["postings"] = json.loads(json.dumps(pair))
        vouchers.append(voucher_args)
    return vouchers if len(vouchers) > 1 else None


async def _prepare_voucher_postings(client: TripletexClient, args: dict, ctx: EntityContext | None) -> dict | None:
    if "postings" not in args or not isinstance(args["postings"], list):
        return None
    _normalize_year_end_depreciation_postings(args["postings"], args.get("description"), ctx)
    _normalize_year_end_prepaid_reversal_postings(args["postings"], args.get("description"), ctx)
    _normalize_year_end_tax_provision_postings(args["postings"], args.get("description"), ctx)
    correction_error = _validate_ledger_error_correction_postings(args["postings"], args.get("description"), ctx)
    if correction_error is not None:
        return correction_error
    _normalize_month_end_closing_pair_postings(args["postings"], args.get("description"), ctx)
    is_paid_receipt = _looks_like_paid_receipt_voucher(ctx, args["postings"])
    for i, posting in enumerate(args["postings"]):
        if isinstance(posting, dict):
            posting.pop("guiRow", None)
            posting["row"] = i + 1
            account = _get_cached_account(ctx, posting) or {}
            account_vat_id = _extract_reference_id(account.get("vatType"))
            legal_vat_ids = {
                vat_id
                for vat_id in (
                    _extract_reference_id(vat)
                    for vat in (account.get("legalVatTypes") or [])
                )
                if vat_id is not None
            }
            current_vat_id = _extract_reference_id(posting.get("vatType"))
            preferred_vat_id = _preferred_account_vat_id(
                account,
                current_vat_id,
                ctx,
                prefer_account_default=is_paid_receipt and posting.get("amountGross", 0) > 0,
            )
            no_vat_only = account_vat_id == 0 and not any(vat_id != 0 for vat_id in legal_vat_ids)
            if no_vat_only:
                if posting.pop("vatType", None) is not None:
                    logger.info("Removed vatType from voucher posting because the account is VAT-locked")
            elif (
                is_paid_receipt
                and posting.get("amountGross", 0) > 0
                and preferred_vat_id is not None
                and current_vat_id != preferred_vat_id
            ):
                posting["vatType"] = {"id": preferred_vat_id}
                logger.info(
                    f"Normalized receipt voucher vatType to account-supported id={preferred_vat_id}"
                )
            elif (
                account.get("vatLocked")
                and posting.get("amountGross", 0) > 0
                and preferred_vat_id is not None
                and current_vat_id != preferred_vat_id
            ):
                posting["vatType"] = {"id": preferred_vat_id}
                logger.info(
                    f"Normalized locked voucher vatType to account-supported id={preferred_vat_id}"
                )
            elif current_vat_id is not None and legal_vat_ids and current_vat_id not in legal_vat_ids:
                if preferred_vat_id is not None:
                    posting["vatType"] = {"id": preferred_vat_id}
                    logger.info(
                        f"Replaced invalid voucher vatType id={current_vat_id} with account-supported id={preferred_vat_id}"
                    )
                else:
                    posting.pop("vatType", None)
                    logger.info(
                        f"Removed invalid voucher vatType id={current_vat_id} because the account has no default VAT type"
                    )
            if (
                ctx
                and ctx.last_department_id
                and "department" not in posting
                and posting.get("amountGross", 0) > 0
            ):
                posting["department"] = {"id": ctx.last_department_id}
                logger.info(f"Auto-injected department id={ctx.last_department_id} into voucher posting")
    _normalize_supplier_invoice_posting_references(args["postings"], ctx)
    await _normalize_supplier_invoice_software_account(client, args["postings"], ctx, args.get("description"))
    _normalize_simple_supplier_invoice_amounts(args["postings"], ctx)
    await _expand_simple_supplier_invoice_vat_split(client, args["postings"], ctx)
    total = sum(
        p.get("amountGross", 0) for p in args["postings"] if isinstance(p, dict)
    )
    if abs(total) > 0.01:
        return {
            "error": (
                f"Voucher postings do not balance. Net total: {total}. Debit (positive) and credit (negative) "
                "amounts must sum to zero. Adjust the posting amounts and retry."
            )
        }
    return None


async def _post_voucher_with_retry(client: TripletexClient, args: dict, ctx: EntityContext | None) -> dict:
    try:
        return await client.post("/ledger/voucher", json=args)
    except Exception as e:
        retry_args = json.loads(json.dumps(args))
        retried = False
        error_text = str(e).lower()
        if "mva-kode 0" in error_text or "ingen avgiftsbehandling" in error_text:
            for posting in retry_args.get("postings", []):
                if isinstance(posting, dict) and "vatType" in posting:
                    posting.pop("vatType", None)
                    retried = True
            if retried:
                logger.info("Voucher account is locked to no VAT; retrying without vatType on postings")
        if "leverandÃ¸r mangler" in error_text and ctx and ctx.last_customer_id:
            for posting in retry_args.get("postings", []):
                if (
                    isinstance(posting, dict)
                    and posting.get("amountGross", 0) < 0
                    and "supplier" not in posting
                ):
                    posting["supplier"] = {"id": ctx.last_customer_id}
                    retried = True
                    logger.info(f"Auto-injected supplier id={ctx.last_customer_id} into voucher posting retry")
        if "kunde mangler" in error_text and ctx and ctx.last_customer_id:
            injected_customer = False
            for posting in retry_args.get("postings", []):
                account = _get_cached_account(ctx, posting) if isinstance(posting, dict) else {}
                if (
                    isinstance(posting, dict)
                    and str((account or {}).get("number")) == "1500"
                    and "customer" not in posting
                ):
                    posting["customer"] = {"id": ctx.last_customer_id}
                    retried = True
                    injected_customer = True
                    logger.info(
                        f"Auto-injected customer id={ctx.last_customer_id} into receivables voucher posting retry"
                    )
            for posting in retry_args.get("postings", []):
                if (
                    isinstance(posting, dict)
                    and posting.get("amountGross", 0) > 0
                    and "customer" not in posting
                    and not injected_customer
                ):
                    posting["customer"] = {"id": ctx.last_customer_id}
                    retried = True
                    logger.info(f"Auto-injected customer id={ctx.last_customer_id} into voucher posting retry")
        if retried:
            for i, posting in enumerate(retry_args.get("postings", [])):
                if isinstance(posting, dict):
                    posting["row"] = i + 1
            return await client.post("/ledger/voucher", json=retry_args)
        raise


def _record_timesheet_hours_in_context(
    ctx: EntityContext | None,
    employee_id: int | None,
    project_id: int | None,
    activity_id: int | None,
    entry_date: str | None,
    hours,
) -> None:
    if (
        ctx is None
        or employee_id is None
        or project_id is None
        or activity_id is None
        or not isinstance(entry_date, str)
    ):
        return
    booked_hours = _coerce_number(hours)
    key = (employee_id, project_id, activity_id, entry_date)
    ctx.timesheet_hours_by_day[key] = round(ctx.timesheet_hours_by_day.get(key, 0.0) + booked_hours, 2)


async def _create_timesheet_entries(
    client: TripletexClient,
    args: dict,
    ctx: EntityContext | None,
) -> dict:
    employee_id = _extract_reference_id(args.get("employee"))
    project_id = _extract_reference_id(args.get("project"))
    activity_id = _extract_reference_id(args.get("activity"))
    normalized_date = _normalize_timesheet_date(
        ctx,
        employee_id,
        project_id,
        activity_id,
        args.get("date"),
        args.get("hours"),
    )
    if normalized_date and normalized_date != args.get("date"):
        args["date"] = normalized_date
        logger.info(f"Normalized timesheet entry date to {normalized_date}")
    requested_hours = _coerce_number(args.get("hours"))
    if requested_hours <= 12:
        return await client.post("/timesheet/entry", json=args)
    try:
        candidate = datetime.date.fromisoformat(args["date"])
    except (KeyError, ValueError, TypeError):
        return await client.post("/timesheet/entry", json=args)
    candidate = _next_working_day(candidate)
    remaining_hours = requested_hours
    results: list[dict] = []
    while remaining_hours > 0.01:
        chunk_hours = round(min(8.0, remaining_hours), 2)
        entry_args = json.loads(json.dumps(args))
        entry_args["hours"] = chunk_hours
        entry_args["date"] = _normalize_timesheet_date(
            ctx,
            employee_id,
            project_id,
            activity_id,
            candidate.isoformat(),
            chunk_hours,
            daily_limit=8.0,
        )
        result = await client.post("/timesheet/entry", json=entry_args)
        results.append(result)
        _record_timesheet_hours_in_context(
            ctx,
            employee_id,
            project_id,
            activity_id,
            entry_args.get("date"),
            chunk_hours,
        )
        remaining_hours = round(remaining_hours - chunk_hours, 2)
        candidate = _next_working_day(datetime.date.fromisoformat(entry_args["date"]) + datetime.timedelta(days=1))
    return {
        "value": (results[-1] or {}).get("value", {}),
        "values": [result.get("value", {}) for result in results],
        "_skip_track": True,
    }


def _has_meaningful_search_filters(params: dict) -> bool:
    """Treat unfiltered list requests as unsafe to avoid random matches."""
    for key, value in params.items():
        if key in PASSIVE_SEARCH_PARAMS:
            continue
        if value in (None, "", [], {}):
            continue
        return True
    return False


def _preferred_customer_id(ctx: EntityContext | None) -> int | None:
    if ctx is None:
        return None
    return ctx.last_sales_customer_id or ctx.last_customer_id


def _preferred_supplier_id(ctx: EntityContext | None) -> int | None:
    if ctx is None:
        return None
    return ctx.last_supplier_id or ctx.last_customer_id


def _preferred_project_manager_id(ctx: EntityContext | None) -> int | None:
    if ctx is None:
        return None
    if len(ctx.employee_ids or []) > 1:
        return ctx.employee_ids[0]
    return ctx.last_employee_id


def _normalize_timesheet_date(
    ctx: EntityContext | None,
    employee_id: int | None,
    project_id: int | None,
    activity_id: int | None,
    requested_date: str | None,
    hours,
    *,
    daily_limit: float = 24.0,
) -> str | None:
    if (
        ctx is None
        or employee_id is None
        or project_id is None
        or activity_id is None
        or not isinstance(requested_date, str)
    ):
        return requested_date
    try:
        candidate = datetime.date.fromisoformat(requested_date)
    except ValueError:
        return requested_date
    try:
        requested_hours = float(hours)
    except (TypeError, ValueError):
        return requested_date
    project_start_raw = ctx.project_start_dates.get(project_id)
    if project_start_raw:
        try:
            project_start = datetime.date.fromisoformat(project_start_raw)
            if candidate < project_start:
                candidate = project_start
        except ValueError:
            pass
    if requested_hours <= 0 or requested_hours > daily_limit:
        return candidate.isoformat()
    for _ in range(366):
        key = (employee_id, project_id, activity_id, candidate.isoformat())
        used_hours = ctx.timesheet_hours_by_day.get(key, 0.0)
        if used_hours + requested_hours <= daily_limit:
            return candidate.isoformat()
        candidate += datetime.timedelta(days=1)
    return requested_date


def _next_working_day(candidate: datetime.date) -> datetime.date:
    while candidate.weekday() >= 5:
        candidate += datetime.timedelta(days=1)
    return candidate


def _employee_snapshot_full_name(snapshot: dict | None) -> str:
    if not isinstance(snapshot, dict):
        return ""
    first = str(snapshot.get("firstName") or "").strip()
    last = str(snapshot.get("lastName") or "").strip()
    return " ".join(part for part in (first, last) if part).strip()


def _find_employee_id_by_email(ctx: EntityContext | None, email: str | None) -> int | None:
    if ctx is None or not ctx.employee_snapshots:
        return None
    target = str(email or "").strip().lower()
    if not target:
        return None
    for employee_id, snapshot in ctx.employee_snapshots.items():
        if str(snapshot.get("email") or "").strip().lower() == target:
            return employee_id
    return None


def _find_employee_id_by_name(ctx: EntityContext | None, name: str | None) -> int | None:
    if ctx is None or not ctx.employee_snapshots:
        return None
    target = str(name or "").strip().lower()
    if not target:
        return None
    for employee_id, snapshot in ctx.employee_snapshots.items():
        if _employee_snapshot_full_name(snapshot).lower() == target:
            return employee_id
    return None


def _extract_prompt_timesheet_assignments(ctx: EntityContext | None) -> list[dict]:
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return []
    pattern = re.compile(
        r"([^\W\d_][\w'’.-]*(?:\s+[^\W\d_][\w'’.-]*)+)\s*\(([^)]*)\)\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:timer|hours?|hrs?)",
        re.IGNORECASE,
    )
    assignments: list[dict] = []
    for match in pattern.finditer(raw_text):
        metadata = match.group(2) or ""
        email_match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", metadata)
        hours = _coerce_number(str(match.group(3)).replace(",", "."))
        if hours <= 0:
            continue
        assignments.append(
            {
                "name": str(match.group(1) or "").strip(),
                "email": email_match.group(1) if email_match else None,
                "hours": round(hours, 2),
            }
        )
    return assignments


def _extract_prompt_total_timesheet_hours(ctx: EntityContext | None) -> float | None:
    assignments = _extract_prompt_timesheet_assignments(ctx)
    if assignments:
        return round(sum(_coerce_number(item.get("hours")) for item in assignments), 2)
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    matches = re.findall(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:timer|hours?|hrs?)", raw_text, re.IGNORECASE)
    if not matches:
        return None
    total = round(sum(_coerce_number(match.replace(",", ".")) for match in matches), 2)
    return total if total > 0 else None


def _resolve_timesheet_employee_from_prompt(args: dict, ctx: EntityContext | None) -> int | None:
    explicit_email = args.get("employeeEmail") or args.get("email")
    employee_id = _find_employee_id_by_email(ctx, explicit_email)
    if employee_id is not None:
        return employee_id
    explicit_name = args.get("employeeName") or args.get("name")
    employee_id = _find_employee_id_by_name(ctx, explicit_name)
    if employee_id is not None:
        return employee_id
    if ctx is None or len(ctx.employee_ids or []) <= 1:
        return None
    requested_hours = round(_coerce_number(args.get("hours")), 2)
    if requested_hours <= 0:
        return None
    matches = [
        assignment
        for assignment in _extract_prompt_timesheet_assignments(ctx)
        if abs(_coerce_number(assignment.get("hours")) - requested_hours) <= 0.01
    ]
    resolved_ids: list[int] = []
    for assignment in matches:
        assignment_employee_id = _find_employee_id_by_email(ctx, assignment.get("email"))
        if assignment_employee_id is None:
            assignment_employee_id = _find_employee_id_by_name(ctx, assignment.get("name"))
        if assignment_employee_id is not None and assignment_employee_id not in resolved_ids:
            resolved_ids.append(assignment_employee_id)
    if len(resolved_ids) == 1:
        return resolved_ids[0]
    return None


async def _ensure_project_activity_link(
    client: TripletexClient,
    ctx: EntityContext | None,
    project_id: int | None,
    activity_id: int | None,
) -> None:
    prompt_text = _normalize_prompt_text((ctx.prompt_text if ctx else None) or "")
    if (
        ctx is None
        or project_id is None
        or activity_id is None
        or not any(
            token in prompt_text
            for token in (
                "prosjektsyklus",
                "projectcycle",
                "projectinvoice",
                "kundefaktura",
                "invoice",
                "fakturer",
                "timesheetplusprojectinvoice",
            )
        )
        or (project_id, activity_id) in ctx.linked_project_activity_pairs
    ):
        return
    payload = {"project": {"id": project_id}, "activity": {"id": activity_id}}
    budget_amount = _extract_prompt_budget_amount(ctx)
    if budget_amount is not None:
        payload["budgetFeeCurrency"] = round(budget_amount, 2)
    try:
        logger.info(
            "Auto-linking project id=%s and activity id=%s before timesheet entry",
            project_id,
            activity_id,
        )
        result = await client.post("/project/projectActivity", json=payload)
        ctx.track("create_project_activity", result, payload)
    except Exception as exc:
        logger.warning(
            "Project/activity auto-link failed for project id=%s activity id=%s; continuing with timesheet entry: %s",
            project_id,
            activity_id,
            exc,
        )


def _normalize_prompt_text(value: str | None) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def _prompt_contains_any_email(ctx: EntityContext | None) -> bool:
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return False
    return re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", raw_text, re.IGNORECASE) is not None


def _looks_like_placeholder_email(email: str | None) -> bool:
    text = str(email or "").strip().lower()
    if "@" not in text:
        return False
    domain = text.split("@", 1)[1]
    return domain in {"example.org", "example.com", "example.net", "example.invalid", "test.invalid"}


def _prompt_is_contract_or_onboarding_task(ctx: EntityContext | None) -> bool:
    normalized = _normalize_prompt_text((ctx.prompt_text if ctx else None) or "")
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (
            "arbeidskontrakt",
            "employmentcontract",
            "joboffer",
            "offerletter",
            "tilbudsbrev",
            "onboarding",
            "contratodetrabajo",
            "cartadeoferta",
            "completelaincorporacion",
            "createemployeewithallthedat",
            "creaelempleado",
        )
    )


def _prompt_describes_budget_not_fixed_price(ctx: EntityContext | None) -> bool:
    normalized = _normalize_prompt_text((ctx.prompt_text if ctx else None) or "")
    if not normalized:
        return False
    budget_tokens = ("budget", "budsjett", "orcamento", "presupuesto")
    fixed_price_tokens = ("fixedprice", "fastpris", "prixfixe", "precofixo", "forfait")
    return any(token in normalized for token in budget_tokens) and not any(
        token in normalized for token in fixed_price_tokens
    )


def _parse_localized_amount_token(token: str | None) -> float | None:
    text = str(token or "").strip().replace("\u00a0", " ")
    if not text:
        return None
    text = re.sub(r"\s+", "", text).replace("'", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if text.count(",") == 1 and len(text.split(",")[-1]) in (1, 2):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "." in text and text.count(".") > 1:
        text = text.replace(".", "")
    elif "." in text and len(text.split(".")[-1]) > 2:
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def _extract_prompt_fixed_price_amount(ctx: EntityContext | None) -> float | None:
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    normalized_text = unicodedata.normalize("NFKD", raw_text.lower())
    normalized_text = "".join(ch for ch in normalized_text if not unicodedata.combining(ch))
    match = re.search(
        r"(?:fixed\s*price|fastpris|prix\s*fixe|preco\s*fixo|forfait)\D{0,25}([0-9][0-9\s\u00a0.,']*)\s*(?:nok|kr)\b",
        normalized_text,
    )
    if not match:
        return None
    amount = _parse_localized_amount_token(match.group(1))
    return round(amount, 2) if amount and amount > 0 else None


def _prompt_mentions_fixed_price(ctx: EntityContext | None) -> bool:
    return _extract_prompt_fixed_price_amount(ctx) not in (None, 0)


def _prompt_mentions_milestone_invoice(ctx: EntityContext | None) -> bool:
    normalized = _normalize_prompt_text((ctx.prompt_text if ctx else None) or "")
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (
            "milestone",
            "stagepayment",
            "paymentbystage",
            "pagamentoporetapa",
            "pagamentodeetapa",
            "etapa",
            "delbetaling",
            "tranche",
        )
    )


def _extract_prompt_milestone_invoice_fraction(ctx: EntityContext | None) -> float | None:
    if not _prompt_mentions_milestone_invoice(ctx):
        return None
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    normalized_text = unicodedata.normalize("NFKD", raw_text.lower())
    normalized_text = "".join(ch for ch in normalized_text if not unicodedata.combining(ch))
    match = re.search(
        r"([0-9]+(?:[.,][0-9]+)?)\s*(?:%|percent|prosent|porcento|pourcent)",
        normalized_text,
    )
    if not match:
        return None
    percentage = _coerce_number(str(match.group(1)).replace(",", "."))
    if percentage <= 0:
        return None
    return round(percentage / 100.0, 4) if percentage > 1 else round(percentage, 4)


def _prompt_mentions_partial_payments(ctx: EntityContext | None) -> bool:
    normalized = _normalize_prompt_text((ctx.prompt_text if ctx else None) or "")
    if not normalized:
        return False
    partial_tokens = (
        "partialpayment",
        "partialpayments",
        "pagoparcial",
        "pagosparciales",
        "delbetaling",
        "delbetalinger",
        "teilzahlung",
        "teilzahlungen",
        "paiementpartiel",
        "paiementspartiels",
    )
    return any(token in normalized for token in partial_tokens)


def _parse_iso_date(value: str | None) -> datetime.date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _iso_date_plus_days(value: str | None, days: int) -> str | None:
    parsed = _parse_iso_date(value)
    if parsed is None:
        return None
    return (parsed + datetime.timedelta(days=days)).isoformat()


def _normalize_undated_travel_window(
    departure_date: str | None,
    return_date: str | None,
    ctx: EntityContext | None,
) -> tuple[str | None, str | None]:
    if ctx is None or _prompt_has_explicit_calendar_date(ctx):
        return departure_date, return_date
    departure = _parse_iso_date(departure_date)
    return_parsed = _parse_iso_date(return_date)
    if departure is None or return_parsed is None or return_parsed < departure:
        return departure_date, return_date
    if departure.weekday() < 5:
        return departure_date, return_date
    normalized_departure = _next_working_day(departure)
    shift_days = (normalized_departure - departure).days
    normalized_return = return_parsed + datetime.timedelta(days=shift_days)
    return normalized_departure.isoformat(), normalized_return.isoformat()


def _date_within_category(target: datetime.date | None, from_date: str | None, to_date: str | None) -> bool:
    if target is None:
        return True
    from_parsed = _parse_iso_date(from_date)
    to_parsed = _parse_iso_date(to_date)
    if from_parsed is not None and target < from_parsed:
        return False
    if to_parsed is not None and target >= to_parsed:
        return False
    return True


def _infer_per_diem_country_code(args: dict, ctx: EntityContext | None) -> str | None:
    explicit = args.get("countryCode")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().upper()
    normalized = re.sub(
        r"[^a-z0-9]+",
        "",
        _normalize_occupation_name(
        " ".join(
            part
            for part in (
                str(args.get("location") or "").strip(),
                str((ctx.prompt_text if ctx else None) or "").strip(),
            )
            if part
        ),
        ),
    )
    if not normalized:
        return None
    norwegian_markers = (
        "norge",
        "norway",
        "tromso",
        "oslo",
        "bergen",
        "trondheim",
        "stavanger",
        "bodo",
        "alesund",
        "kirkenes",
        "innenriks",
    )
    if any(marker in normalized for marker in norwegian_markers):
        return "NO"
    return None


def _infer_travel_destination(args: dict, ctx: EntityContext | None) -> str | None:
    sources = [
        args.get("location"),
        args.get("destination"),
        args.get("title"),
        (ctx.prompt_text if ctx else None),
    ]
    location_markers = (
        ("tromso", "Tromsø"),
        ("oslo", "Oslo"),
        ("bergen", "Bergen"),
        ("trondheim", "Trondheim"),
        ("stavanger", "Stavanger"),
        ("bodo", "Bodø"),
        ("alesund", "Ålesund"),
        ("kirkenes", "Kirkenes"),
    )
    for source in sources:
        if not source:
            continue
        normalized = re.sub(r"[^a-z0-9]+", "", _normalize_occupation_name(str(source)))
        for marker, display in location_markers:
            if marker in normalized:
                return display
    return None


def _prompt_mentions_per_diem(ctx: EntityContext | None) -> bool:
    text = ((ctx.prompt_text if ctx else None) or "").strip()
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "", _normalize_occupation_name(text))
    markers = (
        "perdiem",
        "dailyallowance",
        "indemnitejournaliere",
        "indemnitesjournalieres",
        "tagegeld",
        "tagessatz",
        "daggodt",
        "diett",
        "ajudadecusto",
        "ajudasdecusto",
        "taxadiaria",
        "taxasdiarias",
    )
    return any(marker in normalized for marker in markers)


def _classify_travel_cost_kind(value: str | None) -> str | None:
    normalized = _normalize_occupation_name(value)
    if not normalized:
        return None
    flight_markers = (
        "flight",
        "airfare",
        "airticket",
        "airticket",
        "plane",
        "fly",
        "flug",
        "bilhetedeaviao",
        "billetdavion",
        "flybillett",
        "flyreise",
        "boardingpass",
    )
    taxi_markers = ("taxi", "drosje", "cab", "uber")
    if any(marker in normalized for marker in flight_markers):
        return "flight"
    if any(marker in normalized for marker in taxi_markers):
        return "taxi"
    return None


def _travel_cost_category_text(category: dict) -> str:
    account = category.get("account") or {}
    parts = [
        category.get("displayName"),
        category.get("description"),
        account.get("name"),
        account.get("number"),
    ]
    return _normalize_occupation_name(" ".join(str(part) for part in parts if part not in (None, "")))


def _select_travel_cost_category(categories: list[dict], comments: str | None) -> dict | None:
    kind = _classify_travel_cost_kind(comments)
    if kind is None:
        return None
    specific_markers = {
        "flight": (
            "flight",
            "airfare",
            "airticket",
            "airticket",
            "plane",
            "fly",
            "flug",
            "flybillett",
            "flyreise",
        ),
        "taxi": ("taxi", "drosje", "cab", "uber"),
    }[kind]
    generic_markers = ("transport", "reise", "travel")
    best: tuple[int, dict] | None = None
    for category in categories:
        if not isinstance(category, dict) or category.get("id") is None:
            continue
        text = _travel_cost_category_text(category)
        if not text:
            continue
        score = 0
        if any(marker in text for marker in specific_markers):
            score += 100
        if any(marker in text for marker in generic_markers):
            score += 20
        if best is None or score > best[0]:
            best = (score, category)
    if best is None or best[0] <= 0:
        return None
    return best[1]


def _prompt_has_explicit_calendar_date(ctx: EntityContext | None) -> bool:
    text = ((ctx.prompt_text if ctx else None) or "").strip()
    if not text:
        return False
    return bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        or re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text)
    )


def _infer_default_travel_cost_date(args: dict, ctx: EntityContext | None) -> str | None:
    if ctx is None:
        return None
    departure_date = ctx.last_travel_expense_departure_date
    return_date = ctx.last_travel_expense_return_date
    kind = _classify_travel_cost_kind(args.get("comments"))
    if kind == "flight":
        return departure_date
    if kind == "taxi":
        return return_date or departure_date
    return None


def _pick_per_diem_rate_category(
    values: list[dict],
    departure_date: str | None,
    return_date: str | None,
    country_code: str | None,
    overnight_accommodation: str | None,
) -> dict | None:
    departure = _parse_iso_date(departure_date)
    return_date_parsed = _parse_iso_date(return_date)
    normalized_country = (country_code or "").strip().upper()
    overnight = str(overnight_accommodation or "").strip().upper()
    best: tuple[int, datetime.date, dict] | None = None

    for item in values:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        if str(item.get("type") or "").upper() not in ("", "PER_DIEM"):
            continue
        if normalized_country == "NO" and item.get("isValidDomestic") is False:
            continue
        if normalized_country and normalized_country != "NO" and item.get("isValidForeignTravel") is False:
            continue
        if overnight and overnight != "NONE" and item.get("isValidAccommodation") is False:
            continue
        if not _date_within_category(departure, item.get("fromDate"), item.get("toDate")):
            continue
        if not _date_within_category(return_date_parsed, item.get("fromDate"), item.get("toDate")):
            continue

        score = 0
        if normalized_country == "NO" and item.get("isValidDomestic") is True:
            score += 40
        elif normalized_country and normalized_country != "NO" and item.get("isValidForeignTravel") is True:
            score += 40
        if overnight and overnight != "NONE" and item.get("isValidAccommodation") is True:
            score += 20
        if overnight and overnight != "NONE" and item.get("isRequiresOvernightAccommodation") is True:
            score += 10
        category_from = _parse_iso_date(item.get("fromDate")) or datetime.date.min
        if departure and category_from <= departure:
            score += 5
        candidate = (score, category_from, item)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    return best[2] if best else None


async def _resolve_per_diem_rate_category(
    client: TripletexClient,
    args: dict,
    ctx: EntityContext | None,
) -> dict | None:
    departure_date = args.get("departureDate") or (ctx.last_travel_expense_departure_date if ctx else None)
    return_date = args.get("returnDate") or (ctx.last_travel_expense_return_date if ctx else None)
    country_code = _infer_per_diem_country_code(args, ctx)
    overnight = str(args.get("overnightAccommodation") or "").strip().upper()
    params: dict[str, object] = {"type": "PER_DIEM", "count": 100}
    if departure_date:
        params["dateFrom"] = departure_date
    return_exclusive = _iso_date_plus_days(return_date, 1)
    if return_exclusive:
        params["dateTo"] = return_exclusive
    if overnight and overnight != "NONE":
        params["isValidAccommodation"] = True
    if country_code == "NO":
        params["isValidDomestic"] = True
    logger.info(
        "Resolving per diem rate category with params=%s, departure=%s, return=%s, country=%s",
        params,
        departure_date,
        return_date,
        country_code or "",
    )
    result = await client.get("/travelExpense/rateCategory", params=params)
    values = result.get("values", [])
    selected = _pick_per_diem_rate_category(values, departure_date, return_date, country_code, overnight)
    if selected is None:
        logger.warning(
            "No matching per diem rate category found for departure=%s return=%s country=%s overnight=%s",
            departure_date,
            return_date,
            country_code or "",
            overnight or "",
        )
        return None
    selected_id = selected.get("id")
    if ctx is not None and selected_id is not None:
        ctx.last_rate_category_id = selected_id
    logger.info(
        "Resolved per diem rate category id=%s name=%s fromDate=%s toDate=%s",
        selected_id,
        selected.get("name", ""),
        selected.get("fromDate"),
        selected.get("toDate"),
    )
    return {"id": selected_id} if selected_id is not None else None


def _extract_prompt_budget_amount(ctx: EntityContext | None) -> float | None:
    if not _prompt_describes_budget_not_fixed_price(ctx):
        return None
    raw_text = (ctx.prompt_text if ctx else None) or ""
    if not raw_text:
        return None
    normalized_text = unicodedata.normalize("NFKD", raw_text.lower())
    normalized_text = "".join(ch for ch in normalized_text if not unicodedata.combining(ch))
    match = re.search(
        r"(?:budget|budsjett|orcamento|presupuesto)\D{0,25}([0-9][0-9\s\u00a0.,']*)",
        normalized_text,
    )
    if not match:
        return None
    token = match.group(1).strip()
    if not token:
        return None
    token = token.replace("\u00a0", "").replace(" ", "").replace("'", "")
    if token.count(",") and token.count("."):
        last_comma = token.rfind(",")
        last_dot = token.rfind(".")
        decimal_separator = "," if last_comma > last_dot else "."
        thousands_separator = "." if decimal_separator == "," else ","
        token = token.replace(thousands_separator, "")
        if decimal_separator == ",":
            token = token.replace(",", ".")
    elif token.count(",") == 1 and token.count(".") == 0:
        left, right = token.split(",", 1)
        token = f"{left}.{right}" if len(right) <= 2 else f"{left}{right}"
    elif token.count(".") > 1:
        token = token.replace(".", "")
    elif token.count(".") == 1:
        left, right = token.split(".", 1)
        if len(right) > 2:
            token = f"{left}{right}"
    amount = _coerce_number(token)
    return amount if amount > 0 else None


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


def _extract_reference_id(value) -> int | None:
    if isinstance(value, dict):
        value = value.get("id")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


WORKING_HOURS_SCHEME_VALUES = {
    "NOT_SHIFT",
    "ROUND_THE_CLOCK",
    "SHIFT_365",
    "OFFSHORE_336",
    "CONTINUOUS",
    "OTHER_SHIFT",
    "NOT_CHOSEN",
}


def _normalize_identifier(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    normalized = re.sub(r"\D+", "", str(value))
    return normalized or None


def _normalize_code_text(value: str | int | None) -> str:
    if value in (None, ""):
        return ""
    return re.sub(r"[^0-9A-Za-z]+", "", str(value)).upper()


def _normalize_occupation_name(value: str | None) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip().lower()
    text = (
        text.replace("æ", "ae")
        .replace("ø", "o")
        .replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def _occupation_name_tokens(value: str | None) -> list[str]:
    if value in (None, ""):
        return []
    text = str(value).strip().lower()
    text = (
        text.replace("Ã¦", "ae")
        .replace("Ã¸", "o")
        .replace("Ã¥", "a")
        .replace("Ã¤", "a")
        .replace("Ã¶", "o")
        .replace("Ã¼", "u")
        .replace("ÃŸ", "ss")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]


def _find_occupation_code_by_name(
    values: list[dict],
    occupation_name: str | None,
    *,
    min_score: int = 1,
) -> dict | None:
    normalized_name = _normalize_occupation_name(occupation_name)
    tokens = _occupation_name_tokens(occupation_name)
    token_set = set(tokens)
    best: tuple[int, int, dict] | None = None

    for item in values:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        candidate_name = str(item.get("nameNO") or "")
        candidate_normalized = _normalize_occupation_name(candidate_name)
        candidate_tokens = set(_occupation_name_tokens(candidate_name))
        score = 0
        if candidate_normalized == normalized_name and normalized_name:
            score = 300
        elif token_set and candidate_tokens == token_set:
            score = 250
        elif token_set and token_set.issubset(candidate_tokens):
            score = 200
        elif normalized_name and candidate_normalized.startswith(normalized_name):
            score = 150
        elif normalized_name and normalized_name.startswith(candidate_normalized) and candidate_normalized:
            score = 120
        elif token_set and token_set.intersection(candidate_tokens):
            score = 50 + len(token_set.intersection(candidate_tokens))
        if score <= 0:
            continue
        candidate = (score, len(candidate_normalized), item)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if best is None or best[0] < min_score:
        return None
    return best[2]


def _normalize_percentage(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _vat_direction_from_name(name: str) -> str:
    normalized = str(name or "").lower()
    return "incoming" if ("inng" in normalized or "incoming" in normalized) else "outgoing"


def _looks_like_fee_text(*parts) -> bool:
    text = " ".join(str(part or "") for part in parts).lower()
    keywords = (
        "mahn",
        "dunning",
        "reminder fee",
        "late fee",
        "lembrete",
        "taxa de lembrete",
        "taxa lembrete",
        "purre",
        "purring",
        "inkasso",
        "gebyr",
    )
    return any(keyword in text for keyword in keywords)


def _looks_like_software_service_text(*parts) -> bool:
    text = " ".join(str(part or "") for part in parts).lower()
    keywords = (
        "cloud storage",
        "skylagring",
        "saas",
        "software subscription",
        "programvareabonnement",
        "hosting",
        "web hosting",
        "server hosting",
        "software",
        "programvare",
    )
    return any(keyword in text for keyword in keywords)


async def _resolve_vat_type(
    client: TripletexClient,
    ctx: EntityContext | None,
    percentage,
    direction: str = "outgoing",
) -> dict | None:
    normalized_percentage = _normalize_percentage(percentage)
    if normalized_percentage is None:
        return None
    if ctx and ctx.vat_type_cache:
        cached_id = ctx.vat_type_cache.get((normalized_percentage, direction))
        if cached_id is not None:
            return {"id": cached_id}
    try:
        result = await client.get("/ledger/vatType", params={
            "percentage": str(int(normalized_percentage) if normalized_percentage.is_integer() else normalized_percentage),
            "fields": "id,number,name,percentage",
        })
        _track_lookup_context(ctx, "/ledger/vatType", result)
        values = result.get("values", [])
        candidates = []
        for item in values:
            item_percentage = _normalize_percentage(item.get("percentage"))
            if item.get("id") is None or item_percentage != normalized_percentage:
                continue
            item_direction = _vat_direction_from_name(item.get("name", ""))
            score = 10 if item_direction == direction else 0
            if normalized_percentage == 0 and direction == "outgoing":
                if item.get("id") == 0:
                    score += 5
                if str(item.get("number", "")).strip() == "0":
                    score += 4
            candidates.append((score, item))
        if candidates:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            return {"id": candidates[0][1]["id"]}
    except Exception as e:
        logger.warning(f"Failed to resolve VAT type for {normalized_percentage}% ({direction}): {e}")
    return None


def _pick_occupation_code(values: list[dict], raw_code: str | int | None) -> dict | None:
    normalized_requested = _normalize_code_text(raw_code)
    if not normalized_requested:
        return None
    exact_matches = []
    prefix_matches = []
    for item in values:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        normalized_candidate = _normalize_code_text(item.get("code"))
        if not normalized_candidate:
            continue
        if normalized_candidate == normalized_requested:
            exact_matches.append(item)
        elif normalized_candidate.startswith(normalized_requested) or normalized_requested.startswith(normalized_candidate):
            prefix_matches.append(item)
    if exact_matches:
        return exact_matches[0]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def _get_cached_account(ctx: EntityContext | None, posting: dict | None) -> dict | None:
    if ctx is None or not ctx.account_cache or not isinstance(posting, dict):
        return None
    account_id = _extract_reference_id(posting.get("account"))
    if account_id is None:
        return None
    return ctx.account_cache.get(account_id)


def _preferred_account_vat_id(
    account: dict,
    current_vat_id: int | None,
    ctx: EntityContext | None,
    *,
    prefer_account_default: bool = False,
) -> int | None:
    account_vat_id = _extract_reference_id(account.get("vatType"))
    legal_vat_ids = {
        vat_id
        for vat_id in (
            _extract_reference_id(vat)
            for vat in (account.get("legalVatTypes") or [])
        )
        if vat_id is not None
    }
    if prefer_account_default and account_vat_id not in (None, 0):
        if not legal_vat_ids or account_vat_id in legal_vat_ids:
            return account_vat_id
    if current_vat_id is not None and current_vat_id in legal_vat_ids and current_vat_id != 0:
        return current_vat_id
    if account_vat_id not in (None, 0) and (not legal_vat_ids or account_vat_id in legal_vat_ids):
        return account_vat_id
    if ctx and ctx.last_vat_type_id in legal_vat_ids:
        return ctx.last_vat_type_id
    legal_non_zero = sorted(vat_id for vat_id in legal_vat_ids if vat_id != 0)
    if legal_non_zero:
        return legal_non_zero[0]
    return None


def _looks_like_paid_receipt_voucher(ctx: EntityContext | None, postings: list[dict] | None) -> bool:
    if ctx is None or not isinstance(postings, list):
        return False
    for posting in postings:
        if not isinstance(posting, dict):
            continue
        if _coerce_number(posting.get("amountGross")) >= 0:
            continue
        account = _get_cached_account(ctx, posting) or {}
        if str(account.get("number")) == "1920" or account.get("isBankAccount") is True:
            return True
    return False


def _normalize_simple_supplier_invoice_amounts(postings: list[dict] | None, ctx: EntityContext | None) -> bool:
    if not isinstance(postings, list):
        return False
    positive_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    negative_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(positive_postings) != 1 or len(negative_postings) != 1:
        return False
    expense_posting = positive_postings[0]
    payable_posting = negative_postings[0]
    if _extract_reference_id(expense_posting.get("vatType")) in (None, 0):
        return False
    payable_account = _get_cached_account(ctx, payable_posting) or {}
    is_supplier_liability = "supplier" in payable_posting or str(payable_account.get("number")) == "2400"
    if not is_supplier_liability:
        return False
    expense_amount = round(_coerce_number(expense_posting.get("amountGross")), 2)
    payable_amount = round(abs(_coerce_number(payable_posting.get("amountGross"))), 2)
    if payable_amount <= expense_amount + 0.01:
        return False
    expense_posting["amountGross"] = payable_amount
    payable_currency_amount = abs(_coerce_number(payable_posting.get("amountGrossCurrency", payable_amount)))
    expense_posting["amountGrossCurrency"] = round(payable_currency_amount, 2)
    logger.info(
        f"Normalized supplier invoice expense posting from net {expense_amount} to gross {payable_amount} based on accounts payable amount"
    )
    return True


def _normalize_supplier_invoice_posting_references(postings: list[dict] | None, ctx: EntityContext | None) -> bool:
    if not isinstance(postings, list):
        return False
    payable_candidates = []
    for posting in postings:
        if not isinstance(posting, dict):
            continue
        if _coerce_number(posting.get("amountGross")) >= 0:
            continue
        account = _get_cached_account(ctx, posting) or {}
        if "supplier" in posting or str(account.get("number")) == "2400":
            payable_candidates.append(posting)
    if len(payable_candidates) != 1:
        return False
    payable_posting = payable_candidates[0]
    removed = 0
    for posting in postings:
        if not isinstance(posting, dict) or posting is payable_posting:
            continue
        if posting.pop("supplier", None) is not None:
            removed += 1
    if removed:
        logger.info(f"Removed supplier reference from {removed} non-payables supplier-invoice posting(s)")
        return True
    return False


def _find_cached_vat_percentage(ctx: EntityContext | None, vat_type_id: int | None, direction: str | None = None) -> float | None:
    if ctx is None or vat_type_id is None or not ctx.vat_type_cache:
        return None
    for (percentage, cached_direction), cached_id in ctx.vat_type_cache.items():
        if cached_id != vat_type_id:
            continue
        if direction is None or cached_direction == direction:
            return float(percentage)
    for (percentage, _cached_direction), cached_id in ctx.vat_type_cache.items():
        if cached_id == vat_type_id:
            return float(percentage)
    return None


async def _normalize_supplier_invoice_software_account(
    client: TripletexClient,
    postings: list[dict] | None,
    ctx: EntityContext | None,
    voucher_description: str | None = None,
) -> bool:
    if not isinstance(postings, list):
        return False
    positive_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    negative_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(positive_postings) != 1 or len(negative_postings) != 1:
        return False
    expense_posting = positive_postings[0]
    payable_posting = negative_postings[0]
    payable_account = _get_cached_account(ctx, payable_posting) or {}
    is_supplier_liability = "supplier" in payable_posting or str(payable_account.get("number")) == "2400"
    if not is_supplier_liability:
        return False
    expense_account = _get_cached_account(ctx, expense_posting) or {}
    if str(expense_account.get("number")) != "6340":
        return False
    if not _looks_like_software_service_text(
        voucher_description,
        expense_posting.get("description"),
        payable_posting.get("description"),
    ):
        return False
    software_account = _find_cached_account_by_number(ctx, "6420")
    if software_account is None:
        result = await client.get(
            "/ledger/account",
            params={
                "number": "6420",
                "fields": "id,number,name,vatType,vatLocked,requiresDepartment,legalVatTypes,isApplicableForSupplierInvoice,isBankAccount",
            },
        )
        values = result.get("values", [])
        if values:
            software_account = values[0]
            if ctx is not None:
                ctx.account_cache[software_account["id"]] = software_account
    if software_account is None:
        return False
    expense_posting["account"] = {"id": software_account["id"]}
    logger.info("Normalized supplier invoice software/cloud expense account from 6340 to 6420")
    return True


async def _expand_simple_supplier_invoice_vat_split(
    client: TripletexClient,
    postings: list[dict] | None,
    ctx: EntityContext | None,
) -> bool:
    if not isinstance(postings, list):
        return False
    positive_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) > 0]
    negative_postings = [p for p in postings if isinstance(p, dict) and _coerce_number(p.get("amountGross")) < 0]
    if len(positive_postings) != 1 or len(negative_postings) != 1:
        return False
    expense_posting = positive_postings[0]
    payable_posting = negative_postings[0]
    vat_type_id = _extract_reference_id(expense_posting.get("vatType"))
    if vat_type_id in (None, 0):
        return False
    payable_account = _get_cached_account(ctx, payable_posting) or {}
    is_supplier_liability = "supplier" in payable_posting or str(payable_account.get("number")) == "2400"
    if not is_supplier_liability:
        return False
    gross_amount = round(abs(_coerce_number(payable_posting.get("amountGross"))), 2)
    expense_amount = round(_coerce_number(expense_posting.get("amountGross")), 2)
    if abs(expense_amount - gross_amount) > 0.01:
        return False
    vat_percentage = _find_cached_vat_percentage(ctx, vat_type_id, direction="incoming")
    if vat_percentage in (None, 0):
        return False
    vat_account = _find_cached_account_by_number(ctx, "2710")
    if vat_account is None:
        result = await client.get(
            "/ledger/account",
            params={
                "number": "2710",
                "fields": "id,number,name,vatType,vatLocked,requiresDepartment,legalVatTypes,isApplicableForSupplierInvoice,isBankAccount",
            },
        )
        values = result.get("values", [])
        if values:
            vat_account = values[0]
            if ctx is not None:
                ctx.account_cache[vat_account["id"]] = vat_account
    if vat_account is None:
        return False
    vat_amount = round(gross_amount * vat_percentage / (100 + vat_percentage), 2)
    net_amount = round(gross_amount - vat_amount, 2)
    if vat_amount <= 0 or net_amount <= 0:
        return False
    expense_posting["amountGross"] = net_amount
    expense_posting["amountGrossCurrency"] = net_amount
    expense_posting.pop("vatType", None)
    vat_posting = {
        "account": {"id": vat_account["id"]},
        "amountGross": vat_amount,
        "amountGrossCurrency": vat_amount,
        "description": f"Inngående MVA {vat_percentage:g}%",
    }
    if ctx and ctx.last_department_id:
        vat_posting["department"] = {"id": ctx.last_department_id}
    insert_index = postings.index(expense_posting) + 1
    postings.insert(insert_index, vat_posting)
    for i, posting in enumerate(postings):
        if isinstance(posting, dict):
            posting["row"] = i + 1
    logger.info(
        f"Expanded supplier invoice gross amount {gross_amount} into net {net_amount} + VAT {vat_amount} on account 2710"
    )
    return True


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
        skip_track = False
        if isinstance(result, dict):
            skip_track = bool(result.pop("_skip_track", False))
        if ctx is not None:
            if not skip_track:
                ctx.track(name, result, args)
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


async def _ensure_project_manager(client: TripletexClient) -> int | None:
    """Find a project manager from an existing project."""
    try:
        result = await client.get("/project", params={"fields": "id,projectManager", "count": 50})
        values = result.get("values", [])
        for project in values:
            project_manager = project.get("projectManager") or {}
            project_manager_id = project_manager.get("id")
            if project_manager_id:
                logger.info(f"Reusing existing project manager id={project_manager_id} from project search")
                return project_manager_id
    except Exception as e:
        logger.warning(f"Failed to find reusable project manager: {e}")
    return None


async def _resolve_employment_id(client: TripletexClient, args: dict, ctx: EntityContext | None) -> int | None:
    employment_id = _extract_reference_id(args.get("employment"))
    if employment_id is None:
        employment_id = _extract_reference_id(args.get("employmentId"))
    if employment_id is not None:
        return employment_id
    if ctx and ctx.last_employment_id:
        return ctx.last_employment_id
    employee_id = _extract_reference_id(args.get("employee"))
    if employee_id is None:
        employee_id = _extract_reference_id(args.get("employeeId"))
    if employee_id is None and ctx and ctx.last_employee_id:
        employee_id = ctx.last_employee_id
    if employee_id is None:
        return None
    try:
        result = await client.get("/employee/employment", params={
            "employeeId": employee_id,
            "fields": "id,startDate,endDate",
            "count": 20,
        })
        values = result.get("values", [])
        if not values:
            return None
        target_date = args.get("date") or args.get("fromDate") or args.get("startDate")
        if target_date:
            try:
                target = datetime.date.fromisoformat(target_date)
                matching = []
                for employment in values:
                    start_date_raw = employment.get("startDate")
                    if not isinstance(start_date_raw, str):
                        continue
                    try:
                        start_date = datetime.date.fromisoformat(start_date_raw)
                    except ValueError:
                        continue
                    end_date_raw = employment.get("endDate")
                    end_date = None
                    if isinstance(end_date_raw, str) and end_date_raw:
                        try:
                            end_date = datetime.date.fromisoformat(end_date_raw)
                        except ValueError:
                            end_date = None
                    if start_date <= target and (end_date is None or target <= end_date):
                        matching.append((start_date, employment))
                if matching:
                    matching.sort(key=lambda item: item[0], reverse=True)
                    employment_id = matching[0][1].get("id")
                    if employment_id is not None:
                        return employment_id
            except ValueError:
                for employment in values:
                    if employment.get("startDate") == target_date and employment.get("id") is not None:
                        return employment["id"]
        return values[0].get("id")
    except Exception as e:
        logger.warning(f"Failed to resolve employment ID from employee: {e}")
        return None


async def _get_employee_snapshot(client: TripletexClient, employee_id: int) -> dict:
    result = await client.get(
        f"/employee/{employee_id}",
        params={"fields": "id,firstName,lastName,email,dateOfBirth,department"},
    )
    return result.get("value") or {}


async def _ensure_employment_for_employee(
    client: TripletexClient,
    employee_id: int,
    effective_date: str,
    ctx: EntityContext | None,
    *,
    skip_existing_lookup: bool = False,
) -> int | None:
    if not skip_existing_lookup:
        employment_id = await _resolve_employment_id(
            client,
            {"employeeId": employee_id, "date": effective_date},
            ctx,
        )
        if employment_id is not None:
            if ctx is not None:
                ctx.last_employment_id = employment_id
                ctx.last_employee_id = employee_id
            return employment_id

    employee = await _get_employee_snapshot(client, employee_id)
    date_of_birth = employee.get("dateOfBirth")
    if not date_of_birth:
        placeholder_date_of_birth = "1990-01-01"
        logger.warning(
            f"Employee {employee_id} is missing dateOfBirth; applying placeholder {placeholder_date_of_birth} to create employment"
        )
        await client.put(
            f"/employee/{employee_id}",
            json={"id": employee_id, "dateOfBirth": placeholder_date_of_birth},
        )
        date_of_birth = placeholder_date_of_birth

    result = await client.post(
        "/employee/employment",
        json={
            "employee": {"id": employee_id, "dateOfBirth": date_of_birth},
            "startDate": effective_date,
        },
    )
    value = result.get("value") or {}
    employment_id = value.get("id")
    if employment_id is not None and ctx is not None:
        ctx.last_employment_id = employment_id
        ctx.last_employee_id = employee_id
    return employment_id


async def _resolve_employee_id(
    client: TripletexClient,
    args: dict,
    ctx: EntityContext | None,
    employment_id: int | None = None,
) -> int | None:
    employee_id = _extract_reference_id(args.get("employee"))
    if employee_id is None:
        employee_id = _extract_reference_id(args.get("employeeId"))
    if employee_id is not None:
        return employee_id
    if ctx and ctx.last_employee_id:
        return ctx.last_employee_id
    if employment_id is None:
        employment_id = _extract_reference_id(args.get("employment"))
    if employment_id is None:
        employment_id = _extract_reference_id(args.get("employmentId"))
    if employment_id is None and ctx and ctx.last_employment_id:
        employment_id = ctx.last_employment_id
    if employment_id is None:
        return None
    try:
        result = await client.get(f"/employee/employment/{employment_id}")
        employee = (result.get("value") or {}).get("employee") or {}
        return _extract_reference_id(employee)
    except Exception as e:
        logger.warning(f"Failed to resolve employee ID from employment {employment_id}: {e}")
        return None


async def _resolve_working_hours_scheme(client: TripletexClient, scheme_value) -> str | None:
    if isinstance(scheme_value, dict):
        if scheme_value.get("workingHoursScheme") in WORKING_HOURS_SCHEME_VALUES:
            return scheme_value["workingHoursScheme"]
        scheme_value = scheme_value.get("id")
    if scheme_value in (None, ""):
        return None
    if isinstance(scheme_value, str):
        normalized = scheme_value.strip().upper()
        if normalized in WORKING_HOURS_SCHEME_VALUES:
            return normalized
        if normalized.isdigit():
            scheme_value = int(normalized)
        else:
            try:
                result = await client.get("/employee/employment/workingHoursScheme", params={
                    "fields": "id,workingHoursScheme,nameNO,code",
                    "count": 50,
                })
                values = result.get("values", [])
                for item in values:
                    candidates = {
                        str(item.get("workingHoursScheme", "")).strip().upper(),
                        str(item.get("nameNO", "")).strip().upper(),
                        str(item.get("code", "")).strip().upper(),
                    }
                    if normalized in candidates:
                        return item.get("workingHoursScheme")
            except Exception as e:
                logger.warning(f"Failed to resolve working-hours scheme {scheme_value}: {e}")
            return None
    try:
        scheme_id = int(scheme_value)
    except (TypeError, ValueError):
        return None
    try:
        result = await client.get("/employee/employment/workingHoursScheme", params={
            "id": str(scheme_id),
            "fields": "id,workingHoursScheme,nameNO,code",
            "count": 1,
        })
        values = result.get("values", [])
        if values:
            return values[0].get("workingHoursScheme")
    except Exception as e:
        logger.warning(f"Failed to resolve working-hours scheme ID {scheme_id}: {e}")
    return None


async def _resolve_occupation_code(client: TripletexClient, args: dict):
    occupation_code = args.get("occupationCode")
    if isinstance(occupation_code, dict) and occupation_code.get("id"):
        return occupation_code
    stillingskode = args.get("stillingskode")
    occupation_code_code = args.get("occupationCodeCode")
    occupation_code_name = args.get("occupationCodeName")
    if isinstance(occupation_code, dict):
        occupation_code_code = occupation_code_code or occupation_code.get("code")
        occupation_code_name = occupation_code_name or occupation_code.get("nameNO")
    elif isinstance(occupation_code, (str, int)):
        raw_occupation = str(occupation_code).strip()
        if raw_occupation:
            if raw_occupation.isdigit():
                occupation_code_code = occupation_code_code or raw_occupation
            else:
                occupation_code_name = occupation_code_name or raw_occupation
    if stillingskode:
        stillingskode_text = str(stillingskode).strip()
        if stillingskode_text.isdigit():
            occupation_code_code = occupation_code_code or stillingskode_text
        else:
            occupation_code_name = occupation_code_name or stillingskode_text
    if occupation_code_code:
        try:
            result = await client.get("/employee/employment/occupationCode", params={
                "code": str(occupation_code_code),
                "fields": "id,nameNO,code",
                "count": 20,
            })
            values = result.get("values", [])
            match = _pick_occupation_code(values, occupation_code_code)
            if match is not None:
                return {"id": match["id"]}
            if not values:
                fallback_result = await client.get("/employee/employment/occupationCode", params={
                    "fields": "id,nameNO,code",
                    "count": 200,
                })
                fallback_match = _pick_occupation_code(fallback_result.get("values", []), occupation_code_code)
                if fallback_match is not None:
                    return {"id": fallback_match["id"]}
            if values and values[0].get("id") is not None:
                return {"id": values[0]["id"]}
        except Exception as e:
            logger.warning(f"Failed to resolve occupation code {occupation_code_code}: {e}")
    if occupation_code_name:
        try:
            result = await client.get("/employee/employment/occupationCode", params={
                "nameNO": str(occupation_code_name),
                "fields": "id,nameNO,code",
                "count": 20,
            })
            values = result.get("values", [])
            match = _find_occupation_code_by_name(values, occupation_code_name, min_score=120)
            if match is not None:
                logger.info(
                    "Resolved occupation name %s from direct occupationCode search to id=%s code=%s",
                    occupation_code_name,
                    match.get("id"),
                    match.get("code"),
                )
                return {"id": match["id"]}
            if values:
                logger.info(
                    "No exact occupation-name match for %s in direct search results; scanning full occupation code list",
                    occupation_code_name,
                )
            if not values or match is None:
                fallback_result = await client.get("/employee/employment/occupationCode", params={
                    "fields": "id,nameNO,code",
                    "count": 500,
                })
                fallback_values = fallback_result.get("values", [])
                fallback_match = _find_occupation_code_by_name(fallback_values, occupation_code_name, min_score=120)
                if fallback_match is not None:
                    logger.info(
                        "Resolved occupation name %s from fallback occupationCode scan to id=%s code=%s",
                        occupation_code_name,
                        fallback_match.get("id"),
                        fallback_match.get("code"),
                    )
                    return {"id": fallback_match["id"]}
                logger.warning(
                    "Unable to resolve occupation name %s after fallback occupationCode scan",
                    occupation_code_name,
                )
            if values and values[0].get("id") is not None:
                return {"id": values[0]["id"]}
        except Exception as e:
            logger.warning(f"Failed to resolve occupation name {occupation_code_name}: {e}")
    return None


async def _upsert_standard_time(
    client: TripletexClient,
    employee_id: int,
    from_date: str,
    hours_per_day: float,
) -> dict:
    target_hours = round(_coerce_number(hours_per_day), 2)
    payload = {
        "employee": {"id": employee_id},
        "fromDate": from_date,
        "hoursPerDay": target_hours,
    }
    try:
        result = await client.get("/employee/standardTime", params={
            "employeeId": employee_id,
            "fields": "id,fromDate,hoursPerDay",
            "count": 100,
        })
        for standard_time in result.get("values", []):
            if standard_time.get("fromDate") == from_date and standard_time.get("id") is not None:
                existing_hours = round(_coerce_number(standard_time.get("hoursPerDay")), 2)
                if existing_hours == target_hours:
                    logger.info(
                        f"Standard time already matches {target_hours} hours/day from {from_date} for employee {employee_id}"
                    )
                    return {"value": standard_time}
                standard_time_id = standard_time["id"]
                return await client.put(f"/employee/standardTime/{standard_time_id}", json={"id": standard_time_id, **payload})
    except Exception as e:
        logger.warning(f"Failed to look up existing standard time for employee {employee_id}: {e}")
    return await client.post("/employee/standardTime", json=payload)


async def _list_postings_by_date(client: TripletexClient, date_from: str, date_to: str) -> list[dict]:
    """Fetch ledger postings for a date range, following pagination."""
    values: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        result = await client.get("/ledger/postingByDate", params={
            "dateFrom": date_from,
            "dateTo": date_to,
            "from": offset,
            "count": page_size,
        })
        batch = result.get("values", [])
        if not batch:
            break
        values.extend(batch)
        if len(batch) < page_size:
            break
        offset += len(batch)
    return values


def _coerce_number(value) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
        email = str(args.get("email") or "").strip()
        if (
            email
            and _looks_like_placeholder_email(email)
            and _prompt_is_contract_or_onboarding_task(ctx)
            and not _prompt_contains_any_email(ctx)
        ):
            args.pop("email", None)
            logger.info(
                "Removed placeholder employee email %s for contract/onboarding task without literal email in prompt",
                email,
            )
            if args.get("userType") in (None, "", "STANDARD"):
                args["userType"] = "NO_ACCESS"
                logger.info("Normalized employee userType to NO_ACCESS after removing placeholder email")
        if "userType" not in args:
            args["userType"] = "STANDARD" if args.get("email") else "NO_ACCESS"
        national_identity_number = _normalize_identifier(args.get("nationalIdentityNumber"))
        if national_identity_number:
            args["nationalIdentityNumber"] = national_identity_number
        dnumber = _normalize_identifier(args.get("dnumber"))
        if dnumber:
            args["dnumber"] = dnumber
        if "department" not in args and ctx and ctx.last_department_id:
            args["department"] = {"id": ctx.last_department_id}
            logger.info(f"Auto-injected department id={ctx.last_department_id} into employee")
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
                result = await client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email,department"})
                values = result.get("values", [])
                if values:
                    employee = values[0]
                    employee_id = employee["id"]
                    requested_department_id = _extract_reference_id(args.get("department"))
                    current_department_id = _extract_reference_id(employee.get("department"))
                    if requested_department_id and requested_department_id != current_department_id:
                        logger.info(f"Updating existing employee id={employee_id} with department id={requested_department_id}")
                        return await client.put(
                            f"/employee/{employee_id}",
                            json={"id": employee_id, "department": {"id": requested_department_id}},
                        )
                    logger.info(f"Reusing existing employee id={employee_id} for email {email}")
                    return {"value": employee}
            except Exception:
                pass
        try:
            return await client.post("/employee", json=args)
        except Exception as e:
            # Auto-inject department if required
            if "department" in str(e).lower() and "department" not in args:
                logger.info("Employee requires department — auto-finding/creating one")
                dept_id = ctx.last_department_id if ctx and ctx.last_department_id else await _ensure_department(client)
                if dept_id:
                    args["department"] = {"id": dept_id}
                    try:
                        return await client.post("/employee", json=args)
                    except Exception as retry_error:
                        if email and _is_duplicate_error(retry_error, "bruker med denne e-postadressen", "already exists"):
                            result = await client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email,department"})
                            values = result.get("values", [])
                            if values:
                                employee = values[0]
                                logger.info(f"Employee create hit duplicate email; reusing id={employee['id']}")
                                return {"value": employee}
                        raise
            if email and _is_duplicate_error(e, "bruker med denne e-postadressen", "already exists"):
                result = await client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email,department"})
                values = result.get("values", [])
                if values:
                    employee = values[0]
                    logger.info(f"Employee create hit duplicate email; reusing id={employee['id']}")
                    return {"value": employee}
            raise

    if name == "update_employee":
        eid = args.get("employee_id")
        if eid is None and ctx and ctx.last_employee_id:
            eid = ctx.last_employee_id
            logger.info(f"Auto-injected employee id={eid} into update_employee")
        if eid is None:
            raise ValueError("update_employee requires employee_id")
        fields = dict(args.get("fields") or {})
        if "department" not in fields and ctx and ctx.last_department_id:
            fields["department"] = {"id": ctx.last_department_id}
            logger.info(f"Auto-injected department id={ctx.last_department_id} into update_employee")
        if not fields:
            raise ValueError("update_employee requires fields or usable context such as last_department_id")
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
            result = await client.post("/customer", json=args)
            value = result.get("value", {})
            customer_id = value.get("id")
            update_fields = {}
            requested_is_customer = args.get("isCustomer")
            requested_is_supplier = args.get("isSupplier")
            if requested_is_customer is not None and customer_id is not None and requested_is_customer != value.get("isCustomer"):
                update_fields["isCustomer"] = requested_is_customer
            if requested_is_supplier is not None and customer_id is not None and requested_is_supplier != value.get("isSupplier"):
                update_fields["isSupplier"] = requested_is_supplier
            if update_fields:
                try:
                    logger.info(f"Correcting created customer id={customer_id} flags after create")
                    corrected = await client.put(f"/customer/{customer_id}", json={"id": customer_id, **update_fields})
                    corrected_value = dict(corrected.get("value", {}))
                    corrected_value.setdefault("id", customer_id)
                    corrected_value.update(update_fields)
                    return {"value": corrected_value}
                except Exception as update_error:
                    logger.warning(f"Customer post-create flag correction failed for id={customer_id}: {update_error}")
            return result
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
        vat_percentage = args.pop("vatPercentage", None)
        if vat_percentage is None:
            vat_percentage = args.pop("vatRate", None)
        requested_vat_type_id = _extract_reference_id(args.get("vatType"))
        if "vatType" not in args and vat_percentage not in (None, ""):
            resolved_vat_type = await _resolve_vat_type(client, ctx, vat_percentage, direction="outgoing")
            if resolved_vat_type is not None:
                args["vatType"] = resolved_vat_type
                requested_vat_type_id = resolved_vat_type["id"]
                logger.info(f"Resolved product vatType from {vat_percentage}% to id={resolved_vat_type['id']}")
        elif "vatType" not in args and ctx and ctx.last_vat_type_id:
            args["vatType"] = {"id": ctx.last_vat_type_id}
            requested_vat_type_id = ctx.last_vat_type_id
            logger.info(f"Auto-injected last vatType id={ctx.last_vat_type_id} into product")
        elif "vatType" not in args and _looks_like_fee_text(args.get("name"), args.get("number")):
            resolved_vat_type = await _resolve_vat_type(client, ctx, 0, direction="outgoing")
            if resolved_vat_type is not None:
                args["vatType"] = resolved_vat_type
                requested_vat_type_id = resolved_vat_type["id"]
                logger.info(f"Resolved zero-VAT fee product to vatType id={resolved_vat_type['id']}")
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
            result = await client.post("/product", json=args)
            value = dict(result.get("value", {}))
            product_id = value.get("id")
            actual_vat_type_id = _extract_reference_id(value.get("vatType"))
            if (
                requested_vat_type_id == 0
                and product_id is not None
                and actual_vat_type_id not in (None, requested_vat_type_id)
            ):
                try:
                    logger.info(f"Correcting created product id={product_id} vatType after create")
                    corrected = await client.put(
                        f"/product/{product_id}",
                        json={"id": product_id, "vatType": {"id": requested_vat_type_id}},
                    )
                    corrected_value = dict(corrected.get("value", {}))
                    corrected_value.setdefault("id", product_id)
                    corrected_value["vatType"] = {"id": requested_vat_type_id}
                    return {"value": corrected_value}
                except Exception as update_error:
                    logger.warning(f"Product post-create vatType correction failed for id={product_id}: {update_error}")
                    value["vatType"] = {"id": requested_vat_type_id}
                    return {"value": value}
            return result
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
        preferred_customer_id = _preferred_customer_id(ctx)
        if "customer" not in args and preferred_customer_id:
            args["customer"] = {"id": preferred_customer_id}
            logger.info(f"Auto-injected customer id={preferred_customer_id} into order")
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
        if (
            "orderLines" in args
            and isinstance(args["orderLines"], list)
            and len(args["orderLines"]) == 1
            and _prompt_mentions_fixed_price(ctx)
            and _prompt_mentions_milestone_invoice(ctx)
        ):
            fixed_price_amount = _extract_prompt_fixed_price_amount(ctx)
            milestone_fraction = _extract_prompt_milestone_invoice_fraction(ctx)
            if fixed_price_amount not in (None, 0) and milestone_fraction not in (None, 0):
                target_amount = round(fixed_price_amount * milestone_fraction, 2)
                line = args["orderLines"][0]
                count = _coerce_number(line.get("count") or 1)
                if count <= 0:
                    count = 1.0
                    line["count"] = 1
                if "unitPriceExcludingVatCurrency" in line:
                    current_total = round(_coerce_number(line.get("unitPriceExcludingVatCurrency")) * count, 2)
                    if abs(current_total - target_amount) > 0.01:
                        line["unitPriceExcludingVatCurrency"] = round(target_amount / count, 2)
                        logger.info(
                            "Normalized fixed-price milestone order line to %.2f from prompt fixed price %.2f * fraction %.4f",
                            target_amount,
                            fixed_price_amount,
                            milestone_fraction,
                        )
                elif "unitPriceIncludingVatCurrency" in line:
                    current_total = round(_coerce_number(line.get("unitPriceIncludingVatCurrency")) * count, 2)
                    if abs(current_total - target_amount) > 0.01:
                        line["unitPriceIncludingVatCurrency"] = round(target_amount / count, 2)
                        logger.info(
                            "Normalized fixed-price milestone inclusive order line to %.2f from prompt fixed price %.2f * fraction %.4f",
                            target_amount,
                            fixed_price_amount,
                            milestone_fraction,
                        )
                else:
                    line["count"] = count
                    line["unitPriceExcludingVatCurrency"] = target_amount
                    logger.info(
                        "Auto-set fixed-price milestone order line to %.2f from prompt fixed price %.2f * fraction %.4f",
                        target_amount,
                        fixed_price_amount,
                        milestone_fraction,
                    )
        # Auto-inject default vatType (25% = id 3) on order lines missing it
        if "orderLines" in args:
            has_ex_vat = False
            has_inc_vat = False
            for line in args["orderLines"]:
                if _looks_like_fee_text(line.get("description")):
                    resolved_vat_type = await _resolve_vat_type(client, ctx, 0, direction="outgoing")
                    if resolved_vat_type is not None and _extract_reference_id(line.get("vatType")) != resolved_vat_type["id"]:
                        line["vatType"] = resolved_vat_type
                        logger.info(f"Normalized fee order line vatType to id={resolved_vat_type['id']}")
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
        if "fixedprice" not in args and args.get("fixedPrice") not in (None, ""):
            args["fixedprice"] = _coerce_number(args.pop("fixedPrice"))
            logger.info(f"Normalized create_project fixedPrice -> fixedprice ({args['fixedprice']})")
        if "fixedprice" not in args and _prompt_mentions_fixed_price(ctx):
            fixed_price_amount = _extract_prompt_fixed_price_amount(ctx)
            if fixed_price_amount not in (None, 0):
                args["fixedprice"] = fixed_price_amount
                logger.info(f"Auto-inferred create_project fixedprice={fixed_price_amount} from prompt")
        if _prompt_mentions_fixed_price(ctx) and args.get("isFixedPrice") is not True:
            args["isFixedPrice"] = True
            logger.info("Auto-set create_project isFixedPrice=true from prompt")
        if _prompt_describes_budget_not_fixed_price(ctx) and (
            args.get("isFixedPrice") is True or args.get("fixedprice") not in (None, "")
        ):
            args.pop("isFixedPrice", None)
            args.pop("fixedprice", None)
            args.pop("fixedPrice", None)
            logger.info("Removed fixed-price project fields because the prompt described a budget, not a fixed-price project")
        # Auto-inject projectManager if missing
        preferred_manager_id = _preferred_project_manager_id(ctx)
        if "projectManager" not in args and preferred_manager_id:
            args["projectManager"] = {"id": preferred_manager_id}
            logger.info(f"Auto-injected employee id={preferred_manager_id} as projectManager")
        if "projectManager" not in args:
            project_manager_id = await _ensure_project_manager(client)
            if project_manager_id:
                args["projectManager"] = {"id": project_manager_id}
                logger.info(f"Auto-injected reusable project manager id={project_manager_id} into project")
        preferred_customer_id = _preferred_customer_id(ctx)
        if "customer" not in args and not args.get("isInternal") and preferred_customer_id:
            args["customer"] = {"id": preferred_customer_id}
            logger.info(f"Auto-injected customer id={preferred_customer_id} into project")
        if "customer" not in args and "isInternal" not in args:
            args["isInternal"] = True
            logger.info("Auto-set isInternal=true for customerless project")
        if _is_placeholder_project_number(args.get("number")):
            args["number"] = _generate_project_number()
            logger.info(f"Auto-generated project number {args['number']}")
        if "startDate" not in args:
            args["startDate"] = datetime.date.today().isoformat()
        try:
            return await client.post("/project", json=args)
        except Exception as e:
            if "prosjektleder" in str(e).lower() or "projectmanager.id" in str(e).lower():
                retry_manager_id = await _ensure_project_manager(client)
                current_manager_id = (args.get("projectManager") or {}).get("id")
                if retry_manager_id and retry_manager_id != current_manager_id:
                    retry_args = dict(args)
                    retry_args["projectManager"] = {"id": retry_manager_id}
                    logger.info(f"Retrying project create with reusable project manager id={retry_manager_id}")
                    return await client.post("/project", json=retry_args)
            raise

    if name == "create_activity":
        activity_name = args.get("name")
        if activity_name:
            try:
                result = await client.get("/activity", params={"name": activity_name, "fields": "id,name", "count": 10})
                values = result.get("values", [])
                normalized_name = activity_name.strip().lower()
                for activity in values:
                    if str(activity.get("name", "")).strip().lower() == normalized_name:
                        logger.info(f"Reusing existing activity id={activity['id']} for name {activity_name}")
                        return {"value": activity}
            except Exception:
                pass
        return await client.post("/activity", json=args)

    if name == "create_department":
        department_name = args.get("name")
        if department_name:
            try:
                result = await client.get("/department", params={"name": department_name, "fields": "id,name", "count": 10})
                values = result.get("values", [])
                normalized_name = department_name.strip().lower()
                for department in values:
                    if str(department.get("name", "")).strip().lower() == normalized_name:
                        logger.info(f"Reusing existing department id={department['id']} for name {department_name}")
                        return {"value": department}
            except Exception:
                pass
        return await client.post("/department", json=args)

    if name == "create_employment_details":
        employment_id = await _resolve_employment_id(client, args, ctx)
        if employment_id is None:
            employee_id = await _resolve_employee_id(client, args, ctx)
            effective_date = args.get("date") or args.get("fromDate") or args.get("startDate") or datetime.date.today().isoformat()
            if employee_id is not None:
                employment_id = await _ensure_employment_for_employee(
                    client,
                    employee_id,
                    effective_date,
                    ctx,
                    skip_existing_lookup=True,
                )
        if employment_id is None:
            raise ValueError("create_employment_details requires employment/employmentId, or a prior employee/employment in context")
        employee_id = await _resolve_employee_id(client, args, ctx, employment_id=employment_id)
        department_id = _extract_reference_id(args.get("department"))
        if department_id is None:
            department_id = _extract_reference_id(args.get("departmentId"))
        if department_id is None and ctx and ctx.last_department_id:
            department_id = ctx.last_department_id
        if employee_id is not None and department_id is not None:
            await client.put(
                f"/employee/{employee_id}",
                json={"id": employee_id, "department": {"id": department_id}},
            )
            logger.info(f"Updated employee id={employee_id} to department id={department_id}")
        effective_date = args.get("date") or args.get("fromDate") or args.get("startDate") or datetime.date.today().isoformat()
        annual_salary = args.get("annualSalary")
        if annual_salary is None:
            annual_salary = args.get("salary")
        if annual_salary is not None:
            annual_salary = _coerce_number(annual_salary)
        percentage = args.get("percentageOfFullTimeEquivalent")
        if percentage is None:
            percentage = args.get("employmentPercentage")
        if percentage is not None:
            percentage = _coerce_number(percentage)
        hourly_wage = args.get("hourlyWage")
        if hourly_wage is not None:
            hourly_wage = _coerce_number(hourly_wage)
        remuneration_type = args.get("remunerationType")
        if remuneration_type is None:
            if hourly_wage not in (None, 0):
                remuneration_type = "HOURLY_WAGE"
            elif annual_salary not in (None, 0):
                remuneration_type = "MONTHLY_WAGE"
        occupation_code = await _resolve_occupation_code(client, args)
        employment_type = args.get("employmentType")
        if employment_type is None:
            employment_type = "ORDINARY"
        working_hours_scheme = await _resolve_working_hours_scheme(client, args.get("workingHoursScheme"))
        if working_hours_scheme is None:
            working_hours_scheme = await _resolve_working_hours_scheme(client, args.get("workingHoursSchemeId"))
        if working_hours_scheme is None:
            working_hours_scheme = "NOT_SHIFT"
        payload = _compact_dict({
            "employment": {"id": employment_id},
            "date": effective_date,
            "employmentType": employment_type,
            "employmentForm": args.get("employmentForm"),
            "remunerationType": remuneration_type,
            "workingHoursScheme": working_hours_scheme,
            "shiftDurationHours": args.get("shiftDurationHours"),
            "occupationCode": occupation_code,
            "percentageOfFullTimeEquivalent": percentage,
            "annualSalary": annual_salary,
            "hourlyWage": hourly_wage,
            "payrollTaxMunicipalityId": args.get("payrollTaxMunicipalityId"),
        })
        try:
            result = await client.get("/employee/employment/details", params={
                "employmentId": str(employment_id),
                "fields": "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme",
                "count": 100,
            })
            for details in result.get("values", []):
                if details.get("date") == effective_date and details.get("id") is not None:
                    details_id = details["id"]
                    result = await client.put(f"/employee/employment/details/{details_id}", json={"id": details_id, **payload})
                    break
            else:
                result = await client.post("/employee/employment/details", json=payload)
        except Exception:
            result = await client.post("/employee/employment/details", json=payload)
        hours_per_day = args.get("hoursPerDay")
        if hours_per_day in (None, "") and args.get("hoursPerWeek") not in (None, ""):
            hours_per_day = _coerce_number(args["hoursPerWeek"]) / 5
        elif hours_per_day not in (None, ""):
            hours_per_day = _coerce_number(hours_per_day)
        if employee_id is not None and hours_per_day not in (None, ""):
            await _upsert_standard_time(client, employee_id, effective_date, hours_per_day)
            logger.info(f"Updated employee id={employee_id} standard time to {hours_per_day} hours/day from {effective_date}")
        return result

    if name == "create_standard_time":
        employee_id = await _resolve_employee_id(client, args, ctx)
        if employee_id is None:
            raise ValueError("create_standard_time requires employee/employeeId, or a prior employee in context")
        from_date = args.get("fromDate") or args.get("date") or args.get("startDate") or datetime.date.today().isoformat()
        hours_per_day = args.get("hoursPerDay")
        if hours_per_day in (None, "") and args.get("hoursPerWeek") not in (None, ""):
            hours_per_day = _coerce_number(args["hoursPerWeek"]) / 5
        elif hours_per_day not in (None, ""):
            hours_per_day = _coerce_number(hours_per_day)
        if hours_per_day in (None, ""):
            raise ValueError("create_standard_time requires hoursPerDay or hoursPerWeek")
        return await _upsert_standard_time(client, employee_id, from_date, hours_per_day)

    if name == "create_travel_expense":
        # Auto-inject employee if missing
        if "employee" not in args and ctx and ctx.last_employee_id:
            args["employee"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} into travel expense")
        # Move date fields into nested travelDetails object
        travel_details = args.pop("travelDetails", {})
        if not isinstance(travel_details, dict):
            travel_details = {}
        for date_field in ("departureDate", "returnDate", "departureDateTime", "returnDateTime"):
            val = args.pop(date_field, None)
            if val:
                # Normalize field names
                normalized = date_field.replace("DateTime", "Date")
                travel_details[normalized] = val
        inferred_destination = _infer_travel_destination(args, ctx)
        if inferred_destination and not travel_details.get("destination"):
            travel_details["destination"] = inferred_destination
            logger.info("Auto-inferred travel expense destination=%s", inferred_destination)
        title = str(args.get("title") or "").strip()
        if title and not travel_details.get("purpose"):
            travel_details["purpose"] = title
            logger.info("Auto-inferred travel expense purpose from title")
        departure_date = travel_details.get("departureDate")
        return_date = travel_details.get("returnDate")
        original_departure_date = departure_date
        original_return_date = return_date
        normalized_departure_date, normalized_return_date = _normalize_undated_travel_window(
            departure_date,
            return_date,
            ctx,
        )
        if (
            normalized_departure_date != departure_date
            or normalized_return_date != return_date
        ):
            travel_details["departureDate"] = normalized_departure_date
            travel_details["returnDate"] = normalized_return_date
            departure_date = normalized_departure_date
            return_date = normalized_return_date
            logger.info(
                "Shifted undated travel expense window from departure=%s return=%s to next working window departure=%s return=%s",
                original_departure_date,
                original_return_date,
                normalized_departure_date,
                normalized_return_date,
            )
        departure_parsed = _parse_iso_date(departure_date)
        return_parsed = _parse_iso_date(return_date)
        if (
            departure_parsed is not None
            and return_parsed is not None
            and return_parsed > departure_parsed
            and "isDayTrip" not in travel_details
        ):
            travel_details["isDayTrip"] = False
            logger.info("Auto-set travel expense isDayTrip=false for multi-day trip")
        if _prompt_mentions_per_diem(ctx) and "isCompensationFromRates" not in travel_details:
            travel_details["isCompensationFromRates"] = True
            logger.info("Auto-set travel expense isCompensationFromRates=true from prompt")
        domestic_country = _infer_per_diem_country_code(
            {"location": inferred_destination or travel_details.get("destination") or title},
            ctx,
        )
        if domestic_country == "NO" and "isForeignTravel" not in travel_details:
            travel_details["isForeignTravel"] = False
            logger.info("Auto-set travel expense isForeignTravel=false for domestic trip")
        if travel_details:
            args["travelDetails"] = travel_details
            if ctx is not None:
                ctx.last_travel_expense_departure_date = travel_details.get("departureDate")
                ctx.last_travel_expense_return_date = travel_details.get("returnDate")
                ctx.travel_cost_count = 0
                logger.info(
                    "Tracked travel expense dates departure=%s return=%s for per diem resolution",
                    ctx.last_travel_expense_departure_date,
                    ctx.last_travel_expense_return_date,
                )
        return await client.post("/travelExpense", json=args)

    if name == "create_per_diem_compensation":
        if "travelExpense" not in args and ctx and ctx.last_travel_expense_id:
            args["travelExpense"] = {"id": ctx.last_travel_expense_id}
            logger.info(f"Auto-injected travel expense id={ctx.last_travel_expense_id} into per diem compensation")
        inferred_country = _infer_per_diem_country_code(args, ctx)
        explicit_country = str(args.get("countryCode") or "").strip().upper()
        if explicit_country == "NO" and inferred_country == "NO":
            args.pop("countryCode", None)
            logger.info("Omitted optional domestic countryCode=NO from per diem payload")
        elif "countryCode" not in args and inferred_country and inferred_country != "NO":
            args["countryCode"] = inferred_country
            logger.info(f"Inferred per diem countryCode={inferred_country} from location/prompt")
        if "rateCategory" not in args:
            resolved_rate_category = await _resolve_per_diem_rate_category(client, args, ctx)
            if resolved_rate_category is not None:
                args["rateCategory"] = resolved_rate_category
            elif ctx and ctx.last_rate_category_id:
                args["rateCategory"] = {"id": ctx.last_rate_category_id}
                logger.info(f"Auto-injected fallback rate category id={ctx.last_rate_category_id} into per diem compensation")
        current_args = json.loads(json.dumps(args))
        attempted_signatures: set[str] = set()
        while True:
            attempted_signatures.add(json.dumps(current_args, sort_keys=True))
            try:
                return await client.post("/travelExpense/perDiemCompensation", json=current_args)
            except Exception as e:
                error_text = str(e).lower()
                retry_args = json.loads(json.dumps(current_args))
                retried = False
                if "country not enabled for travel expense" in error_text:
                    if "countryCode" not in retry_args:
                        inferred_country = _infer_per_diem_country_code(retry_args, ctx)
                        if inferred_country:
                            retry_args["countryCode"] = inferred_country
                            retried = True
                            logger.info(
                                "Per diem compensation failed on country validation; retrying with inferred countryCode=%s",
                                inferred_country,
                            )
                    elif str(retry_args.get("countryCode") or "").strip().upper() == "NO":
                        retry_args.pop("countryCode", None)
                        retried = True
                        logger.info(
                            "Per diem compensation failed with countryCode=NO; retrying without optional countryCode",
                        )
                if "satskategori" in error_text or "ratecategory.id" in error_text:
                    resolved_rate_category = await _resolve_per_diem_rate_category(client, retry_args, ctx)
                    if resolved_rate_category is not None and resolved_rate_category != retry_args.get("rateCategory"):
                        retry_args["rateCategory"] = resolved_rate_category
                        retried = True
                        logger.info(
                            "Per diem compensation failed on rate-category validation; retrying with rateCategory id=%s",
                            resolved_rate_category.get("id"),
                        )
                retry_signature = json.dumps(retry_args, sort_keys=True)
                if retried and retry_signature not in attempted_signatures:
                    current_args = retry_args
                    continue
                if retried:
                    logger.warning("Per diem compensation retry payload already attempted; aborting further retries")
                raise

    if name == "create_travel_cost":
        if "travelExpense" not in args and ctx and ctx.last_travel_expense_id:
            args["travelExpense"] = {"id": ctx.last_travel_expense_id}
            logger.info(f"Auto-injected travel expense id={ctx.last_travel_expense_id} into travel cost")
        if "costCategory" not in args and ctx and ctx.last_cost_categories:
            resolved_cost_category = _select_travel_cost_category(ctx.last_cost_categories, args.get("comments"))
            if resolved_cost_category is not None:
                args["costCategory"] = {"id": resolved_cost_category["id"]}
                logger.info(
                    "Resolved travel cost category id=%s from comments=%r",
                    resolved_cost_category["id"],
                    args.get("comments"),
                )
        if "costCategory" not in args and ctx and ctx.last_cost_category_id:
            args["costCategory"] = {"id": ctx.last_cost_category_id}
            logger.info(f"Auto-injected cost category id={ctx.last_cost_category_id} into travel cost")
        if "paymentType" not in args and ctx and ctx.last_payment_type_id:
            args["paymentType"] = {"id": ctx.last_payment_type_id}
            logger.info(f"Auto-injected payment type id={ctx.last_payment_type_id} into travel cost")
        inferred_date = _infer_default_travel_cost_date(args, ctx)
        if args.get("date") in (None, "") and inferred_date:
            args["date"] = inferred_date
            logger.info(
                "Auto-inferred travel cost date=%s from comments=%r",
                inferred_date,
                args.get("comments"),
            )
        elif (
            inferred_date
            and not _prompt_has_explicit_calendar_date(ctx)
            and args.get("date") not in (None, "", inferred_date)
            and _classify_travel_cost_kind(args.get("comments")) in {"flight", "taxi"}
        ):
            previous_date = args.get("date")
            args["date"] = inferred_date
            logger.info(
                "Aligned undated %s travel cost date from %s to %s based on normalized travel expense window",
                _classify_travel_cost_kind(args.get("comments")),
                previous_date,
                inferred_date,
            )
        elif (
            inferred_date
            and ctx
            and ctx.travel_cost_count > 0
            and args.get("date") == ctx.last_travel_expense_departure_date
            and inferred_date != args.get("date")
            and not _prompt_has_explicit_calendar_date(ctx)
            and _classify_travel_cost_kind(args.get("comments")) == "taxi"
        ):
            args["date"] = inferred_date
            logger.info(
                "Adjusted taxi travel cost date to return date=%s for multi-day trip without explicit expense dates",
                inferred_date,
            )
        if (
            args.get("amountCurrencyIncVat") not in (None, "")
            and args.get("count") in (None, "", 1, 1.0)
            and _coerce_number(args.get("rate")) == _coerce_number(args.get("amountCurrencyIncVat"))
        ):
            args.pop("rate", None)
            logger.info("Removed redundant travel cost rate because amountCurrencyIncVat already specifies the amount")
        result = await client.post("/travelExpense/cost", json=args)
        if ctx is not None:
            ctx.travel_cost_count += 1
        return result

    if name == "create_project_activity":
        if (
            "project" not in args
            and "activity" not in args
            and ctx
            and ctx.project_ids
            and ctx.activity_ids
            and ctx.next_project_activity_pair_index < min(len(ctx.project_ids), len(ctx.activity_ids))
        ):
            pair_index = ctx.next_project_activity_pair_index
            args["project"] = {"id": ctx.project_ids[pair_index]}
            args["activity"] = {"id": ctx.activity_ids[pair_index]}
            ctx.next_project_activity_pair_index += 1
            logger.info(f"Auto-injected paired project/activity ids {ctx.project_ids[pair_index]}/{ctx.activity_ids[pair_index]} into project activity")
        if "project" not in args and ctx and ctx.last_project_id:
            args["project"] = {"id": ctx.last_project_id}
            logger.info(f"Auto-injected project id={ctx.last_project_id} into project activity")
        if "activity" not in args and ctx and ctx.last_activity_id:
            args["activity"] = {"id": ctx.last_activity_id}
            logger.info(f"Auto-injected activity id={ctx.last_activity_id} into project activity")
        if (
            args.get("budgetFeeCurrency") in (None, "")
            and args.get("budgetHours") in (None, "")
            and args.get("budgetHourlyRateCurrency") in (None, "")
        ):
            budget_amount = _extract_prompt_budget_amount(ctx)
            if budget_amount is not None:
                args["budgetFeeCurrency"] = round(budget_amount, 2)
                logger.info(f"Auto-injected budgetFeeCurrency={args['budgetFeeCurrency']} into project activity from prompt budget")
        return await client.post("/project/projectActivity", json=args)

    if name == "create_timesheet_entry":
        if "employee" not in args and "employeeId" not in args:
            resolved_employee_id = _resolve_timesheet_employee_from_prompt(args, ctx)
            if resolved_employee_id is not None:
                args["employee"] = {"id": resolved_employee_id}
                logger.info(
                    "Resolved timesheet employee id=%s from prompt hours=%s",
                    resolved_employee_id,
                    args.get("hours"),
                )
        if "employee" not in args and ctx and ctx.last_employee_id:
            args["employee"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} into timesheet entry")
        if "project" not in args and ctx and ctx.last_project_id:
            args["project"] = {"id": ctx.last_project_id}
            logger.info(f"Auto-injected project id={ctx.last_project_id} into timesheet entry")
        if "activity" not in args and ctx and ctx.last_activity_id:
            args["activity"] = {"id": ctx.last_activity_id}
            logger.info(f"Auto-injected activity id={ctx.last_activity_id} into timesheet entry")
        await _ensure_project_activity_link(
            client,
            ctx,
            _extract_reference_id(args.get("project")),
            _extract_reference_id(args.get("activity")),
        )
        return await _create_timesheet_entries(client, args, ctx)

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
        if args.get("fixedRate") in (None, ""):
            budget_amount = _extract_prompt_budget_amount(ctx)
            total_hours = _extract_prompt_total_timesheet_hours(ctx)
            if budget_amount not in (None, 0) and total_hours not in (None, 0):
                args["fixedRate"] = round(budget_amount / total_hours, 2)
                logger.info(
                    "Derived project hourly rate fixedRate=%.2f from budget %.2f / total hours %.2f",
                    args["fixedRate"],
                    budget_amount,
                    total_hours,
                )
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
        split_vouchers = _split_month_end_closing_vouchers(args)
        if split_vouchers:
            results = []
            for voucher_args in split_vouchers:
                validation_error = await _prepare_voucher_postings(client, voucher_args, ctx)
                if validation_error is not None:
                    return validation_error
                results.append(await _post_voucher_with_retry(client, voucher_args, ctx))
            return {
                "value": (results[-1] or {}).get("value", {}),
                "values": [result.get("value", {}) for result in results],
            }
        validation_error = await _prepare_voucher_postings(client, args, ctx)
        if validation_error is not None:
            return validation_error
        return await _post_voucher_with_retry(client, args, ctx)
        if "postings" in args and isinstance(args["postings"], list):
            _normalize_year_end_depreciation_postings(args["postings"], args.get("description"), ctx)
            is_paid_receipt = _looks_like_paid_receipt_voucher(ctx, args["postings"])
            for i, posting in enumerate(args["postings"]):
                if isinstance(posting, dict):
                    posting.pop("guiRow", None)
                    posting["row"] = i + 1
                    account = _get_cached_account(ctx, posting) or {}
                    account_vat_id = _extract_reference_id(account.get("vatType"))
                    legal_vat_ids = {
                        vat_id
                        for vat_id in (
                            _extract_reference_id(vat)
                            for vat in (account.get("legalVatTypes") or [])
                        )
                        if vat_id is not None
                    }
                    current_vat_id = _extract_reference_id(posting.get("vatType"))
                    preferred_vat_id = _preferred_account_vat_id(
                        account,
                        current_vat_id,
                        ctx,
                        prefer_account_default=is_paid_receipt and posting.get("amountGross", 0) > 0,
                    )
                    no_vat_only = account_vat_id == 0 and not any(vat_id != 0 for vat_id in legal_vat_ids)
                    if no_vat_only:
                        if posting.pop("vatType", None) is not None:
                            logger.info("Removed vatType from voucher posting because the account is VAT-locked")
                    elif (
                        is_paid_receipt
                        and posting.get("amountGross", 0) > 0
                        and preferred_vat_id is not None
                        and current_vat_id != preferred_vat_id
                    ):
                        posting["vatType"] = {"id": preferred_vat_id}
                        logger.info(
                            f"Normalized receipt voucher vatType to account-supported id={preferred_vat_id}"
                        )
                    elif (
                        account.get("vatLocked")
                        and posting.get("amountGross", 0) > 0
                        and preferred_vat_id is not None
                        and current_vat_id != preferred_vat_id
                    ):
                        posting["vatType"] = {"id": preferred_vat_id}
                        logger.info(
                            f"Normalized locked voucher vatType to account-supported id={preferred_vat_id}"
                        )
                    elif current_vat_id is not None and legal_vat_ids and current_vat_id not in legal_vat_ids:
                        if preferred_vat_id is not None:
                            posting["vatType"] = {"id": preferred_vat_id}
                            logger.info(
                                f"Replaced invalid voucher vatType id={current_vat_id} with account-supported id={preferred_vat_id}"
                            )
                        else:
                            posting.pop("vatType", None)
                            logger.info(
                                f"Removed invalid voucher vatType id={current_vat_id} because the account has no default VAT type"
                            )
                    if (
                        ctx
                        and ctx.last_department_id
                        and "department" not in posting
                        and posting.get("amountGross", 0) > 0
                    ):
                        posting["department"] = {"id": ctx.last_department_id}
                        logger.info(f"Auto-injected department id={ctx.last_department_id} into voucher posting")
            await _normalize_supplier_invoice_software_account(client, args["postings"], ctx, args.get("description"))
            _normalize_simple_supplier_invoice_amounts(args["postings"], ctx)
            await _expand_simple_supplier_invoice_vat_split(client, args["postings"], ctx)
            # Pre-validate that postings balance before sending
            total = sum(
                p.get("amountGross", 0) for p in args["postings"] if isinstance(p, dict)
            )
            if abs(total) > 0.01:
                return {"error": f"Voucher postings do not balance. Net total: {total}. Debit (positive) and credit (negative) amounts must sum to zero. Adjust the posting amounts and retry."}
        try:
            return await client.post("/ledger/voucher", json=args)
        except Exception as e:
            retry_args = json.loads(json.dumps(args))
            retried = False
            error_text = str(e).lower()
            if "mva-kode 0" in error_text or "ingen avgiftsbehandling" in error_text:
                for posting in retry_args.get("postings", []):
                    if isinstance(posting, dict) and "vatType" in posting:
                        posting.pop("vatType", None)
                        retried = True
                if retried:
                    logger.info("Voucher account is locked to no VAT; retrying without vatType on postings")
            if "leverandør mangler" in error_text and ctx and ctx.last_customer_id:
                for posting in retry_args.get("postings", []):
                    if (
                        isinstance(posting, dict)
                        and posting.get("amountGross", 0) < 0
                        and "supplier" not in posting
                    ):
                        posting["supplier"] = {"id": ctx.last_customer_id}
                        retried = True
                        logger.info(f"Auto-injected supplier id={ctx.last_customer_id} into voucher posting retry")
            if "kunde mangler" in error_text and ctx and ctx.last_customer_id:
                injected_customer = False
                for posting in retry_args.get("postings", []):
                    account = _get_cached_account(ctx, posting) if isinstance(posting, dict) else {}
                    if (
                        isinstance(posting, dict)
                        and str((account or {}).get("number")) == "1500"
                        and "customer" not in posting
                    ):
                        posting["customer"] = {"id": ctx.last_customer_id}
                        retried = True
                        injected_customer = True
                        logger.info(
                            f"Auto-injected customer id={ctx.last_customer_id} into receivables voucher posting retry"
                        )
                for posting in retry_args.get("postings", []):
                    if (
                        isinstance(posting, dict)
                        and posting.get("amountGross", 0) > 0
                        and "customer" not in posting
                        and not injected_customer
                    ):
                        posting["customer"] = {"id": ctx.last_customer_id}
                        retried = True
                        logger.info(f"Auto-injected customer id={ctx.last_customer_id} into voucher posting retry")
            if retried:
                for i, posting in enumerate(retry_args.get("postings", [])):
                    if isinstance(posting, dict):
                        posting["row"] = i + 1
                return await client.post("/ledger/voucher", json=retry_args)
            raise

    if name == "create_salary_transaction":
        try:
            return await client.post("/salary/transaction", json=args)
        except Exception as e:
            error_text = str(e).lower()
            if "arbeidsforhold i perioden" not in error_text and "employment in the period" not in error_text:
                raise
            effective_date = args.get("date")
            if not effective_date:
                year = args.get("year")
                month = args.get("month")
                if year and month:
                    effective_date = f"{int(year):04d}-{int(month):02d}-01"
                else:
                    effective_date = datetime.date.today().isoformat()
            ensured_any = False
            for payslip in args.get("payslips", []):
                if not isinstance(payslip, dict):
                    continue
                employee_id = _extract_reference_id(payslip.get("employee"))
                if employee_id is None:
                    continue
                employment_id = await _ensure_employment_for_employee(client, employee_id, effective_date, ctx)
                if employment_id is not None:
                    ensured_any = True
            if ensured_any:
                logger.info("Created or reused missing employment(s) for salary transaction retry")
                return await client.post("/salary/transaction", json=args)
            raise

    if name == "find_top_expense_account_increases":
        analysis_key = json.dumps(
            {
                "period_a_from": args["period_a_from"],
                "period_a_to": args["period_a_to"],
                "period_b_from": args["period_b_from"],
                "period_b_to": args["period_b_to"],
                "top_n": int(args.get("top_n", 3)),
            },
            sort_keys=True,
        )
        if (
            ctx
            and ctx.last_top_expense_analysis_key == analysis_key
            and isinstance(ctx.last_top_expense_analysis, dict)
        ):
            cached = ctx.last_top_expense_analysis
            return {
                "error": (
                    "find_top_expense_account_increases already ran for this exact comparison. "
                    "Use the topAccounts from the previous result and continue with create_project, "
                    "create_activity, and create_project_activity. Do not call this tool again."
                ),
                "periodA": cached.get("periodA"),
                "periodB": cached.get("periodB"),
                "topAccounts": cached.get("topAccounts", []),
                "nextStepHint": cached.get("nextStepHint"),
            }
        top_n = max(1, int(args.get("top_n", 3)))
        postings_a = await _list_postings_by_date(client, args["period_a_from"], args["period_a_to"])
        postings_b = await _list_postings_by_date(client, args["period_b_from"], args["period_b_to"])

        period_a_totals: dict[int, dict] = {}
        period_b_totals: dict[int, dict] = {}

        def _accumulate(target: dict[int, dict], postings: list[dict]) -> None:
            for posting in postings:
                account = posting.get("account") or {}
                account_id = account.get("id")
                account_number_raw = account.get("number")
                if account_id is None or account_number_raw in (None, ""):
                    continue
                try:
                    account_number = int(str(account_number_raw))
                except ValueError:
                    continue
                if account_number < 4000 or account_number >= 9000:
                    continue
                amount = _coerce_number(posting.get("amount"))
                bucket = target.setdefault(account_id, {
                    "account": {
                        "id": account_id,
                        "number": account_number,
                        "name": account.get("name", ""),
                    },
                    "total": 0.0,
                })
                bucket["total"] += amount

        _accumulate(period_a_totals, postings_a)
        _accumulate(period_b_totals, postings_b)

        combined_ids = set(period_a_totals) | set(period_b_totals)
        accounts: list[dict] = []
        for account_id in combined_ids:
            period_a_total = period_a_totals.get(account_id, {}).get("total", 0.0)
            period_b_total = period_b_totals.get(account_id, {}).get("total", 0.0)
            increase = period_b_total - period_a_total
            account_meta = period_b_totals.get(account_id, {}).get("account") or period_a_totals.get(account_id, {}).get("account")
            accounts.append({
                "account": account_meta,
                "periodAAmount": round(period_a_total, 2),
                "periodBAmount": round(period_b_total, 2),
                "increase": round(increase, 2),
            })

        accounts.sort(key=lambda item: item["increase"], reverse=True)
        top_accounts = [item for item in accounts if item["increase"] > 0][:top_n]
        result = {
            "periodA": {"from": args["period_a_from"], "to": args["period_a_to"]},
            "periodB": {"from": args["period_b_from"], "to": args["period_b_to"]},
            "topAccounts": top_accounts,
            "nextStepHint": "Analysis only. If the task asks for follow-up writes, continue with create_project, create_activity, and create_project_activity for each top account.",
        }
        if ctx is not None:
            ctx.last_top_expense_analysis_key = analysis_key
            ctx.last_top_expense_analysis = result
        return result

    if name == "search_entity":
        entity_type = args["entity_type"]
        params = dict(args.get("params") or {})
        for key, value in args.items():
            if key in {"entity_type", "params"} or value in (None, "", [], {}):
                continue
            params.setdefault(key, value)
        # GET /invoice requires invoiceDateFrom and invoiceDateTo
        if entity_type in {"invoice", "supplierInvoice"}:
            if "invoiceDateFrom" not in params:
                params["invoiceDateFrom"] = "2000-01-01"
                logger.info(f"Auto-injected invoiceDateFrom=2000-01-01 for {entity_type} search")
            if "invoiceDateTo" not in params:
                params["invoiceDateTo"] = "2100-01-01"
                logger.info(f"Auto-injected invoiceDateTo=2100-01-01 for {entity_type} search")
        if not _has_meaningful_search_filters(params):
            logger.warning(f"Blocked unfiltered search_entity call for {entity_type}")
            return {"fullResultSize": 0, "values": []}
        result = await client.get(f"/{entity_type}", params=params)
        # Track first result ID in context for auto-injection
        if ctx:
            values = result.get("values", [])
            if values:
                first = values[0]
                first_id = first.get("id")
                attr_map = {
                    "employee": "last_employee_id",
                    "customer": "last_customer_id",
                    "project": "last_project_id",
                    "invoice": "last_invoice_id",
                    "department": "last_department_id",
                }
                attr = attr_map.get(entity_type)
                if attr and first_id:
                    setattr(ctx, attr, first_id)
                    logger.info(f"EntityContext from search: {attr} = {first_id}")
                if entity_type == "customer" and first_id:
                    if first.get("isCustomer") is not False:
                        ctx.last_sales_customer_id = first_id
                        logger.info(f"EntityContext from search: last_sales_customer_id = {first_id}")
                    if first.get("isSupplier") is True:
                        ctx.last_supplier_id = first_id
                        logger.info(f"EntityContext from search: last_supplier_id = {first_id}")
        return result

    if name == "get_entity":
        return await client.get(f"/{args['entity_type']}/{args['entity_id']}")

    if name == "delete_entity":
        return await client.delete(f"/{args['entity_type']}/{args['entity_id']}")

    if name == "delete_travel_expense":
        te_id = args.get("travel_expense_id")
        if te_id is None and ctx and ctx.last_travel_expense_id:
            te_id = ctx.last_travel_expense_id
        if te_id is None:
            # Search by employee email
            email = args.get("employee_email")
            if not email:
                raise ValueError("delete_travel_expense requires travel_expense_id or employee_email")
            # Find the employee first
            emp_result = await client.get("/employee", params={"email": email, "fields": "id", "count": 1})
            emp_values = emp_result.get("values", [])
            if not emp_values:
                raise ValueError(f"No employee found with email {email}")
            employee_id = emp_values[0]["id"]
            # Search travel expenses for this employee
            te_result = await client.get("/travelExpense", params={
                "employeeId": employee_id,
                "fields": "id,title",
                "count": 100,
            })
            te_values = te_result.get("values", [])
            if not te_values:
                raise ValueError(f"No travel expenses found for employee {email}")
            title = str(args.get("title", "")).strip().lower()
            if title:
                exact_matches = [
                    te for te in te_values
                    if str(te.get("title", "")).strip().lower() == title
                ]
                if len(exact_matches) == 1:
                    te_id = exact_matches[0]["id"]
                elif len(exact_matches) > 1:
                    raise ValueError(f"Multiple travel expenses match title '{args['title']}' for employee {email}; use travel_expense_id")
                else:
                    partial_matches = [
                        te for te in te_values
                        if title in str(te.get("title", "")).strip().lower()
                    ]
                    if len(partial_matches) == 1:
                        te_id = partial_matches[0]["id"]
                    elif len(partial_matches) > 1:
                        raise ValueError(f"Multiple travel expenses partially match title '{args['title']}' for employee {email}; use travel_expense_id")
                    else:
                        raise ValueError(f"No travel expense with title '{args['title']}' found for employee {email}")
            elif len(te_values) == 1:
                te_id = te_values[0]["id"]
            else:
                titles = [str(te.get("title", "")) for te in te_values[:5]]
                raise ValueError(
                    f"Multiple travel expenses found for employee {email}; provide title or travel_expense_id. Candidates: {titles}"
                )
        return await client.delete(f"/travelExpense/{te_id}")

    if name == "reverse_voucher":
        voucher_id = args["voucher_id"]
        date = args["date"]
        return await client.put(f"/ledger/voucher/{voucher_id}/:reverse", params={"date": date})

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
        if path.startswith("/token/session") or path.startswith("/employee/preferences"):
            raise ValueError("Do not use session or logged-in preference endpoints here. Project manager selection is handled automatically by create_project.")
        if path == "/ledger" and method == "GET":
            raise ValueError("Do not use GET /ledger for posting analysis. Use find_top_expense_account_increases or GET /ledger/postingByDate instead.")
        if path == "/ledger/result" and method == "GET":
            raise ValueError("Do not use GET /ledger/result. Use GET /ledger/posting with accountNumberFrom/accountNumberTo over profit-and-loss accounts instead.")
        if method == "GET" and path == "/supplierInvoice/paymentType":
            path = "/ledger/paymentTypeOut"
            logger.info("Normalized unsupported /supplierInvoice/paymentType lookup to /ledger/paymentTypeOut")
        if parsed.query:
            embedded = parse_qs(parsed.query, keep_blank_values=True)
            for k, v in embedded.items():
                if k not in params:
                    params[k] = v[0]  # parse_qs returns lists; take first value
            logger.info(f"Extracted query params from path: {list(embedded.keys())}")
        _normalize_exclusive_date_range(path, params)
        if path.startswith("/ledger/vatType") and isinstance(params.get("fields"), str):
            rewritten_fields = _rewrite_fields_filter(params["fields"], {"rate": "percentage", "direction": ""})
            if rewritten_fields != params["fields"]:
                params["fields"] = rewritten_fields
                logger.info(f"Normalized vatType fields filter to {rewritten_fields}")
        if path.startswith("/ledger/account") and isinstance(params.get("fields"), str):
            enriched_fields = _extend_fields_filter(
                params["fields"],
                [
                    "vatType",
                    "legalVatTypes",
                    "vatLocked",
                    "requiresDepartment",
                    "isApplicableForSupplierInvoice",
                    "isBankAccount",
                ],
            )
            if enriched_fields != params["fields"]:
                params["fields"] = enriched_fields
                logger.info(f"Enriched ledger account fields filter to {enriched_fields}")
        if path == "/ledger/posting":
            if "accountNumber" in params and "accountNumberFrom" not in params and "accountNumberTo" not in params:
                params["accountNumberFrom"] = params["accountNumber"]
                params["accountNumberTo"] = params["accountNumber"]
                params.pop("accountNumber", None)
                logger.info("Normalized ledger/posting accountNumber filter to accountNumberFrom/accountNumberTo")
            if isinstance(params.get("fields"), str):
                rewritten_fields = _rewrite_fields_filter(
                    params["fields"],
                    {
                        "accountingDate": "date",
                    },
                )
                if rewritten_fields != params["fields"]:
                    params["fields"] = rewritten_fields
                    logger.info(f"Normalized ledger/posting fields filter to {rewritten_fields}")
        if path == "/ledger/voucher" and isinstance(params.get("fields"), str):
            rewritten_fields = _normalize_ledger_voucher_fields_filter(params["fields"])
            if rewritten_fields != params["fields"]:
                params["fields"] = rewritten_fields
                logger.info(f"Normalized ledger/voucher fields filter to {rewritten_fields}")
        if path == "/invoice" and isinstance(params.get("fields"), str):
            rewritten_fields = _rewrite_fields_filter(
                params["fields"],
                {
                    "dueDate": "invoiceDueDate",
                    "amountDue": "amountOutstanding",
                    "amountTotal": "amount",
                    "amountRemainder": "amountOutstanding",
                    "amountRemaining": "amountOutstanding",
                    "amountGross": "amount",
                    "isPaid": "",
                    "order": "",
                },
            )
            if rewritten_fields != params["fields"]:
                params["fields"] = rewritten_fields
                logger.info(f"Normalized invoice fields filter to {rewritten_fields}")
        if path == "/invoice" and isinstance(params.get("sorting"), str):
            rewritten_sorting = _rewrite_sorting_filter(
                params["sorting"],
                {
                    "dueDate": "invoiceDueDate",
                },
            )
            if rewritten_sorting != params["sorting"]:
                params["sorting"] = rewritten_sorting
                logger.info(f"Normalized invoice sorting to {rewritten_sorting}")
        if path == "/supplierInvoice" and isinstance(params.get("fields"), str):
            rewritten_fields = _rewrite_fields_filter(
                params["fields"],
                {
                    "dueDate": "invoiceDueDate",
                    "amountDue": "amount",
                    "amountTotal": "amount",
                    "amountRemainder": "amount",
                    "amountRemaining": "amount",
                    "amountOutstanding": "amount",
                    "amountGross": "amount",
                },
            )
            if rewritten_fields != params["fields"]:
                params["fields"] = rewritten_fields
                logger.info(f"Normalized supplierInvoice fields filter to {rewritten_fields}")
        if path == "/product" and "number" in params and "productNumber" not in params:
            number_value = params.get("number")
            if isinstance(number_value, str) and not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", number_value):
                params["productNumber"] = number_value
                params.pop("number", None)
                logger.info(f"Normalized product lookup number -> productNumber for value {number_value}")
        if method == "POST" and path == "/employee/employment/details" and not body:
            translated_args = {}
            for source_key, target_key in {
                "employmentId": "employmentId",
                "employeeId": "employeeId",
                "fromDate": "fromDate",
                "date": "date",
                "salary": "salary",
                "annualSalary": "annualSalary",
                "employmentPercentage": "employmentPercentage",
                "percentageOfFullTimeEquivalent": "percentageOfFullTimeEquivalent",
                "hoursPerDay": "hoursPerDay",
                "hoursPerWeek": "hoursPerWeek",
                "departmentId": "departmentId",
                "workingHoursSchemeId": "workingHoursSchemeId",
                "workingHoursScheme": "workingHoursScheme",
                "remunerationType": "remunerationType",
                "employmentType": "employmentType",
                "employmentForm": "employmentForm",
            }.items():
                if source_key in params:
                    translated_args[target_key] = params[source_key]
            logger.info("Normalized raw /employee/employment/details POST into create_employment_details")
            return await _execute(client, "create_employment_details", translated_args, endpoint_search, ctx)
        if method == "POST" and path == "/employee/standardTime" and not body:
            translated_args = {}
            for source_key, target_key in {
                "employeeId": "employeeId",
                "fromDate": "fromDate",
                "date": "date",
                "startDate": "startDate",
                "hoursPerDay": "hoursPerDay",
                "hoursPerWeek": "hoursPerWeek",
            }.items():
                if source_key in params:
                    translated_args[target_key] = params[source_key]
            logger.info("Normalized raw /employee/standardTime POST into create_standard_time")
            return await _execute(client, "create_standard_time", translated_args, endpoint_search, ctx)
        # Auto-inject required date range for invoice and supplier-invoice list searches.
        list_date_requirements = {
            "/invoice": ("invoiceDateFrom", "invoiceDateTo"),
            "/supplierInvoice": ("invoiceDateFrom", "invoiceDateTo"),
        }
        if method == "GET" and path in list_date_requirements:
            date_from_key, date_to_key = list_date_requirements[path]
            if date_from_key not in params:
                params[date_from_key] = "2000-01-01"
                logger.info(f"Auto-injected {date_from_key} for {path} search")
            if date_to_key not in params:
                params[date_to_key] = "2100-01-01"
                logger.info(f"Auto-injected {date_to_key} for {path} search")
        if method == "PUT" and re.fullmatch(r"/order/\d+/:invoice", path):
            if "invoiceDate" not in params:
                params["invoiceDate"] = datetime.date.today().isoformat()
                logger.info(f"Auto-injected invoiceDate={params['invoiceDate']} for order invoice action")
            await _ensure_bank_account(client)
            logger.info("Preflighted bank account before order invoice action")
        if method == "PUT" and re.fullmatch(r"/supplierInvoice/\d+/:addPayment", path):
            if "paymentType" not in params and "paymentTypeId" in params:
                params["paymentType"] = params.pop("paymentTypeId")
                logger.info(f"Normalized supplier invoice addPayment paymentTypeId -> paymentType ({params['paymentType']})")
            if "amount" not in params and "paidAmount" in params:
                params["amount"] = params.pop("paidAmount")
                logger.info(f"Normalized supplier invoice addPayment paidAmount -> amount ({params['amount']})")
            if "paymentType" not in params:
                params["paymentType"] = 0
                params.setdefault("useDefaultPaymentType", True)
                logger.info("Defaulted supplier invoice addPayment to paymentType=0 with useDefaultPaymentType=true")
            if _prompt_mentions_partial_payments(ctx) and "amount" in params and "partialPayment" not in params:
                params["partialPayment"] = True
                logger.info("Auto-set partialPayment=true for supplier invoice addPayment from prompt context")
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
