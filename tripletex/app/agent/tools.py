"""Tool definitions for OpenAI tool-use and dispatch to Tripletex API."""

import json
import logging

from app.tripletex.client import TripletexClient

logger = logging.getLogger(__name__)


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


TOOL_DEFINITIONS = [
    _tool("create_employee", "Create a new employee in Tripletex.", {
        "type": "object",
        "properties": {
            "firstName": {"type": "string"},
            "lastName": {"type": "string"},
            "email": {"type": "string"},
            "dateOfBirth": {"type": "string", "description": "YYYY-MM-DD"},
            "phoneNumberMobileCountryCode": {"type": "string", "description": "Country code, e.g. +47"},
            "phoneNumberMobile": {"type": "string"},
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
    _tool("create_order", "Create a sales order in Tripletex. Required before creating an invoice.", {
        "type": "object",
        "properties": {
            "customer": {"type": "object", "description": "{\"id\": customer_id}"},
            "orderDate": {"type": "string", "description": "YYYY-MM-DD"},
            "deliveryDate": {"type": "string", "description": "YYYY-MM-DD"},
            "orderLines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "object", "description": "{\"id\": product_id}"},
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
    _tool("create_invoice", "Create an invoice from an order in Tripletex.", {
        "type": "object",
        "properties": {
            "invoiceDate": {"type": "string", "description": "YYYY-MM-DD"},
            "invoiceDueDate": {"type": "string", "description": "YYYY-MM-DD"},
            "customer": {"type": "object", "description": "{\"id\": customer_id}"},
            "orders": {
                "type": "array",
                "items": {"type": "object", "description": "{\"id\": order_id}"},
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


async def dispatch_tool(client: TripletexClient, name: str, args_json: str) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        args = json.loads(args_json)
        result = await _execute(client, name, args)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Tool {name} failed: {e}")
        return json.dumps({"error": str(e)})


async def _execute(client: TripletexClient, name: str, args: dict) -> dict:
    if name == "create_employee":
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
        return await client.post("/product", json=args)

    if name == "create_order":
        return await client.post("/order", json=args)

    if name == "create_invoice":
        return await client.post("/invoice", json=args)

    if name == "create_project":
        return await client.post("/project", json=args)

    if name == "create_department":
        return await client.post("/department", json=args)

    if name == "create_travel_expense":
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
        if method == "GET":
            return await client.get(path, params=params)
        if method == "POST":
            return await client.post(path, json=body)
        if method == "PUT":
            return await client.put(path, json=body)
        if method == "DELETE":
            return await client.delete(path)

    raise ValueError(f"Unknown tool: {name}")
