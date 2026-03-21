import unittest

from app.agent.tools import EntityContext, _execute


class FakeTripletexClient:
    def __init__(self, get_responses=None):
        self.get_responses = get_responses or {}
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        key = (path, tuple(sorted((params or {}).items())))
        return self.get_responses.get(key, {"fullResultSize": 0, "values": []})

    async def post(self, path, json=None):
        self.calls.append(("POST", path, json))
        return {"value": {"id": 999, **(json or {})}}

    async def put(self, path, json=None, params=None):
        self.calls.append(("PUT", path, json, params))
        return {"value": {"id": 999, **(json or {})}}

    async def delete(self, path):
        self.calls.append(("DELETE", path))
        return {}


class ToolRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_employee_updates_existing_by_email(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee",
                    (("email", "mia@example.org"), ("fields", "id,firstName,lastName,email")),
                ): {"fullResultSize": 1, "values": [{"id": 42, "email": "mia@example.org"}]}
            }
        )

        result = await _execute(
            client,
            "create_employee",
            {
                "firstName": "Mia",
                "lastName": "Hoffmann",
                "email": "mia@example.org",
                "startDate": "2026-03-21",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 42)
        self.assertIn(("PUT", "/employee/42", {
            "id": 42,
            "firstName": "Mia",
            "lastName": "Hoffmann",
            "email": "mia@example.org",
            "userType": "STANDARD",
            "employments": [{"startDate": "2026-03-21"}],
        }, None), client.calls)

    async def test_create_customer_updates_existing_supplier(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/customer",
                    (("fields", "id,name,organizationNumber,isCustomer,isSupplier"), ("organizationNumber", "871719500")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 77, "name": "Silberberg GmbH", "isCustomer": True, "isSupplier": False}],
                }
            }
        )

        await _execute(
            client,
            "create_customer",
            {
                "name": "Silberberg GmbH",
                "organizationNumber": "871719500",
                "isCustomer": False,
                "isSupplier": True,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertIn(("PUT", "/customer/77", {
            "id": 77,
            "name": "Silberberg GmbH",
            "organizationNumber": "871719500",
            "isCustomer": False,
            "isSupplier": True,
        }, None), client.calls)

    async def test_search_entity_blocks_unfiltered_searches(self):
        client = FakeTripletexClient()

        result = await _execute(
            client,
            "search_entity",
            {"entity_type": "employee", "params": {}},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result, {"fullResultSize": 0, "values": []})
        self.assertEqual(client.calls, [])

    async def test_create_project_generates_number_and_start_date(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_project",
            {"name": "Plattformintegrasjon", "number": "P001"},
            endpoint_search=None,
            ctx=EntityContext(last_customer_id=100, last_employee_id=200),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/project"))
        self.assertEqual(body["customer"], {"id": 100})
        self.assertEqual(body["projectManager"], {"id": 200})
        self.assertNotEqual(body["number"], "P001")
        self.assertRegex(body["number"], r"^P-\d{8}-[0-9A-F]{6}$")
        self.assertRegex(body["startDate"], r"^\d{4}-\d{2}-\d{2}$")

    async def test_create_order_sets_ex_vat_mode_flag(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_order",
            {
                "orderLines": [
                    {
                        "description": "Consulting",
                        "count": 1,
                        "unitPriceExcludingVatCurrency": 1500,
                    }
                ]
            },
            endpoint_search=None,
            ctx=EntityContext(last_customer_id=321, product_ids=[654]),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/order"))
        self.assertEqual(body["customer"], {"id": 321})
        self.assertEqual(body["orderLines"][0]["product"], {"id": 654})
        self.assertEqual(body["orderLines"][0]["vatType"], {"id": 3})
        self.assertFalse(body["isPrioritizeAmountsIncludingVat"])

    async def test_create_per_diem_uses_last_travel_expense_context(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_per_diem_compensation",
            {
                "rateCategory": {"id": 740},
                "location": "Oslo",
                "overnightAccommodation": "HOTEL",
                "count": 5,
                "rate": 800,
            },
            endpoint_search=None,
            ctx=EntityContext(last_travel_expense_id=555),
        )

        self.assertIn(("POST", "/travelExpense/perDiemCompensation", {
            "travelExpense": {"id": 555},
            "rateCategory": {"id": 740},
            "location": "Oslo",
            "overnightAccommodation": "HOTEL",
            "count": 5,
            "rate": 800,
        }), client.calls)


if __name__ == "__main__":
    unittest.main()
