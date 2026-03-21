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
    last_order_id: int | None = None
    last_employee_id: int | None = None
    last_employment_id: int | None = None
    last_employment_details_id: int | None = None
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
    last_department_id: int | None = None
    last_standard_time_id: int | None = None
    last_voucher_id: int | None = None
    last_vat_type_id: int | None = None
    last_account_id: int | None = None
    account_cache: dict[int, dict] | None = None
    vat_type_cache: dict | None = None  # Maps (percentage, direction_hint) -> vat_type_id
    project_start_dates: dict[int, str] | None = None
    timesheet_hours_by_day: dict[tuple[int, int, int, str], float] | None = None
    last_top_expense_analysis_key: str | None = None
    last_top_expense_analysis: dict | None = None
    next_project_activity_pair_index: int = 0

    def __post_init__(self):
        if self.product_ids is None:
            self.product_ids = []
        if self.project_ids is None:
            self.project_ids = []
        if self.activity_ids is None:
            self.activity_ids = []
        if self.employee_ids is None:
            self.employee_ids = []
        if self.account_cache is None:
            self.account_cache = {}
        if self.project_start_dates is None:
            self.project_start_dates = {}
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
    elif path.startswith("/travelExpense/paymentType") or path.startswith("/invoice/paymentType"):
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
    for raw_part in fields.split(","):
        part = raw_part.strip()
        if not part:
            continue
        normalized = replacements.get(part, part)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        rewritten.append(normalized)
    return ",".join(rewritten)


def _extend_fields_filter(fields: str, extra_fields: list[str]) -> str:
    """Append extra fields to a fields filter if they are missing."""
    base_fields = _rewrite_fields_filter(fields, {})
    current = [field for field in base_fields.split(",") if field]
    seen = set(current)
    for field in extra_fields:
        if field not in seen:
            current.append(field)
            seen.add(field)
    return ",".join(current)


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
    normalized_description = str(description or "").lower()
    if not any(token in normalized_description for token in ("avskriv", "depreciat")):
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
        if "salary accrual" in text or "accrued salary" in text:
            return "salary accrual"
        if "depreciat" in text or "avskriv" in text:
            return "depreciation"
        if "accrual reversal" in text or "prepaid" in text or "forskudds" in text:
            return "accrual reversal"
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
    preferred_prepaid = _find_cached_account_by_number(ctx, "1720")
    text = " ".join(
        [str(description or "")]
        + [
            str(posting.get("description") or "")
            for posting in postings
            if isinstance(posting, dict)
        ]
    ).lower()
    if preferred_prepaid and "1720" in text and str(credit_account.get("number")) != "1720":
        credit_posting["account"] = {"id": preferred_prepaid["id"]}
        logger.info("Normalized accrual-reversal credit posting to account 1720 from cached lookup")


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
    if requested_hours <= 0 or requested_hours > 24:
        return candidate.isoformat()
    for _ in range(366):
        key = (employee_id, project_id, activity_id, candidate.isoformat())
        used_hours = ctx.timesheet_hours_by_day.get(key, 0.0)
        if used_hours + requested_hours <= 24.0:
            return candidate.isoformat()
        candidate += datetime.timedelta(days=1)
    return requested_date


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
        if ctx is not None:
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
            normalized_name = _normalize_occupation_name(occupation_code_name)
            for item in values:
                if _normalize_occupation_name(item.get("nameNO")) == normalized_name and item.get("id") is not None:
                    return {"id": item["id"]}
            if not values:
                fallback_result = await client.get("/employee/employment/occupationCode", params={
                    "fields": "id,nameNO,code",
                    "count": 500,
                })
                fallback_values = fallback_result.get("values", [])
                for item in fallback_values:
                    if _normalize_occupation_name(item.get("nameNO")) == normalized_name and item.get("id") is not None:
                        return {"id": item["id"]}
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
        if "vatType" not in args and vat_percentage not in (None, ""):
            resolved_vat_type = await _resolve_vat_type(client, ctx, vat_percentage, direction="outgoing")
            if resolved_vat_type is not None:
                args["vatType"] = resolved_vat_type
                logger.info(f"Resolved product vatType from {vat_percentage}% to id={resolved_vat_type['id']}")
        elif "vatType" not in args and ctx and ctx.last_vat_type_id:
            args["vatType"] = {"id": ctx.last_vat_type_id}
            logger.info(f"Auto-injected last vatType id={ctx.last_vat_type_id} into product")
        elif "vatType" not in args and _looks_like_fee_text(args.get("name"), args.get("number")):
            resolved_vat_type = await _resolve_vat_type(client, ctx, 0, direction="outgoing")
            if resolved_vat_type is not None:
                args["vatType"] = resolved_vat_type
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
        if path == "/invoice" and isinstance(params.get("fields"), str):
            rewritten_fields = _rewrite_fields_filter(
                params["fields"],
                {
                    "dueDate": "invoiceDueDate",
                    "amountDue": "amountOutstanding",
                    "amountTotal": "amount",
                    "amountRemainder": "amountOutstanding",
                    "amountGross": "amount",
                },
            )
            if rewritten_fields != params["fields"]:
                params["fields"] = rewritten_fields
                logger.info(f"Normalized invoice fields filter to {rewritten_fields}")
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
