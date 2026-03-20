"""Tool definitions for OpenAI tool-use and dispatch to Tripletex API."""

import json
import logging
from dataclasses import dataclass

from app.config import get_settings
from app.endpoint_search import EndpointSearchClient
from app.tripletex.client import TripletexClient

logger = logging.getLogger(__name__)


@dataclass
class EntityContext:
    """Tracks created entity IDs so we can auto-inject missing references.

    GPT-4o frequently omits required entity references (customer on orders,
    orders on invoices) even when told to include them.  This safety net
    fills in the most-recently-created entity when the model forgets.
    """

    last_customer_id: int | None = None
    last_product_id: int | None = None
    last_order_id: int | None = None
    last_employee_id: int | None = None
    last_project_id: int | None = None
    last_invoice_id: int | None = None

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
        }
        attr = mapping.get(name)
        if attr:
            setattr(self, attr, entity_id)
            logger.info(f"EntityContext: {attr} = {entity_id}")


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
    _tool("create_customer", "Create a new customer in Tripletex.", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phoneNumber": {"type": "string"},
            "organizationNumber": {"type": "string"},
            "isCustomer": {"type": "boolean", "default": True},
            "isSupplier": {"type": "boolean"},
            "postalCode": {"type": "string"},
            "city": {"type": "string"},
            "address": {"type": "string"},
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
    _tool("create_order", "Create a sales order. MUST include customer reference from a previous create_customer call. Required before creating an invoice.", {
        "type": "object",
        "properties": {
            "customer": {"type": "object", "description": "REQUIRED — customer reference object, e.g. {\"id\": 123} using the id from create_customer response"},
            "orderDate": {"type": "string", "description": "YYYY-MM-DD"},
            "deliveryDate": {"type": "string", "description": "YYYY-MM-DD"},
            "orderLines": {
                "type": "array",
                "description": "Line items. Each should reference a product OR have a description with price.",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "object", "description": "Product reference {\"id\": product_id} from create_product response"},
                        "description": {"type": "string"},
                        "count": {"type": "number"},
                        "unitPriceExcludingVatCurrency": {"type": "number"},
                        "vatType": {"type": "object"},
                    },
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
            "departureDateTime": {"type": "string"},
            "returnDateTime": {"type": "string"},
        },
        "required": ["employee", "title"],
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
        result = await _execute(client, name, args, endpoint_search=endpoint_search, ctx=ctx)
        if ctx is not None:
            ctx.track(name, result)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Tool {name} failed: {e}")
        return json.dumps({"error": str(e)})


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
        return await client.post("/employee", json=args)

    if name == "update_employee":
        eid = args["employee_id"]
        fields = args["fields"]
        return await client.put(f"/employee/{eid}", json={"id": eid, **fields})

    if name == "create_customer":
        if "isCustomer" not in args:
            args["isCustomer"] = True
        return await client.post("/customer", json=args)

    if name == "update_customer":
        cid = args["customer_id"]
        fields = args["fields"]
        return await client.put(f"/customer/{cid}", json={"id": cid, **fields})

    if name == "create_product":
        try:
            return await client.post("/product", json=args)
        except Exception as e:
            # Handle product number conflict: search for existing product
            error_msg = str(e)
            product_number = args.get("number")
            if "422" in error_msg and product_number:
                logger.info(f"Product number {product_number} conflict, searching for existing product")
                try:
                    result = await client.get("/product", params={"number": product_number, "fields": "id,name,number"})
                    values = result.get("values", [])
                    if values:
                        logger.info(f"Found existing product id={values[0]['id']} for number {product_number}")
                        return {"value": values[0]}
                except Exception:
                    pass
            raise

    if name == "create_order":
        # Auto-inject customer reference if model omitted it
        if "customer" not in args and ctx and ctx.last_customer_id:
            args["customer"] = {"id": ctx.last_customer_id}
            logger.info(f"Auto-injected customer id={ctx.last_customer_id} into order")
        # Auto-inject product reference in orderLines if missing
        if ctx and ctx.last_product_id and "orderLines" in args:
            for line in args["orderLines"]:
                if "product" not in line:
                    line["product"] = {"id": ctx.last_product_id}
                    logger.info(f"Auto-injected product id={ctx.last_product_id} into order line")
        return await client.post("/order", json=args)

    if name == "create_invoice":
        # Auto-inject orders reference if model omitted it
        if "orders" not in args and ctx and ctx.last_order_id:
            args["orders"] = [{"id": ctx.last_order_id}]
            logger.info(f"Auto-injected order id={ctx.last_order_id} into invoice")
        return await client.post("/invoice", json=args)

    if name == "create_project":
        # Auto-inject projectManager if missing
        if "projectManager" not in args and ctx and ctx.last_employee_id:
            args["projectManager"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} as projectManager")
        if "customer" not in args and ctx and ctx.last_customer_id:
            args["customer"] = {"id": ctx.last_customer_id}
            logger.info(f"Auto-injected customer id={ctx.last_customer_id} into project")
        return await client.post("/project", json=args)

    if name == "create_department":
        return await client.post("/department", json=args)

    if name == "create_travel_expense":
        # Auto-inject employee if missing
        if "employee" not in args and ctx and ctx.last_employee_id:
            args["employee"] = {"id": ctx.last_employee_id}
            logger.info(f"Auto-injected employee id={ctx.last_employee_id} into travel expense")
        return await client.post("/travelExpense", json=args)

    if name == "search_entity":
        entity_type = args["entity_type"]
        params = args.get("params", {})
        return await client.get(f"/{entity_type}", params=params)

    if name == "get_entity":
        return await client.get(f"/{args['entity_type']}/{args['entity_id']}")

    if name == "delete_entity":
        return await client.delete(f"/{args['entity_type']}/{args['entity_id']}")

    if name == "tripletex_api_call":
        method = args["method"]
        path = args["path"]
        params = args.get("params")
        body = args.get("body")
        # Guard: POST/PUT without a body causes "Kan ikke være null" errors
        if method in ("POST", "PUT") and not body:
            raise ValueError(f"tripletex_api_call {method} {path} requires a 'body' parameter with the JSON payload")
        if method == "GET":
            return await client.get(path, params=params)
        if method == "POST":
            return await client.post(path, json=body)
        if method == "PUT":
            return await client.put(path, json=body)
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
