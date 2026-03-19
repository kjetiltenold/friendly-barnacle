"""High-level Tripletex API operations.

Each function makes the minimum API calls needed for one logical operation.
These are called by the agent's tool dispatch layer.
"""

from app.tripletex.client import TripletexClient


async def create_employee(client: TripletexClient, **fields) -> dict:
    return await client.post("/employee", json=fields)


async def update_employee(client: TripletexClient, employee_id: int, **fields) -> dict:
    return await client.put(f"/employee/{employee_id}", json={"id": employee_id, **fields})


async def create_customer(client: TripletexClient, **fields) -> dict:
    if "isCustomer" not in fields:
        fields["isCustomer"] = True
    return await client.post("/customer", json=fields)


async def update_customer(client: TripletexClient, customer_id: int, **fields) -> dict:
    return await client.put(f"/customer/{customer_id}", json={"id": customer_id, **fields})


async def create_product(client: TripletexClient, **fields) -> dict:
    return await client.post("/product", json=fields)


async def create_order(client: TripletexClient, **fields) -> dict:
    return await client.post("/order", json=fields)


async def create_invoice(client: TripletexClient, **fields) -> dict:
    return await client.post("/invoice", json=fields)


async def create_project(client: TripletexClient, **fields) -> dict:
    return await client.post("/project", json=fields)


async def create_department(client: TripletexClient, **fields) -> dict:
    return await client.post("/department", json=fields)


async def create_travel_expense(client: TripletexClient, **fields) -> dict:
    return await client.post("/travelExpense", json=fields)


async def delete_travel_expense(client: TripletexClient, expense_id: int) -> dict:
    return await client.delete(f"/travelExpense/{expense_id}")


async def search_entity(client: TripletexClient, entity_type: str, params: dict) -> dict:
    return await client.get(f"/{entity_type}", params=params)


async def get_entity(client: TripletexClient, entity_type: str, entity_id: int) -> dict:
    return await client.get(f"/{entity_type}/{entity_id}")
