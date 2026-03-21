import unittest

from app.agent.tools import EntityContext, _execute


class FakeTripletexClient:
    def __init__(self, get_responses=None, post_errors=None, put_errors=None):
        self.get_responses = get_responses or {}
        self.post_errors = post_errors or {}
        self.put_errors = put_errors or {}
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        key = (path, tuple(sorted((params or {}).items())))
        return self.get_responses.get(key, {"fullResultSize": 0, "values": []})

    async def post(self, path, json=None):
        self.calls.append(("POST", path, json))
        if path in self.post_errors:
            raise self.post_errors[path]
        return {"value": {"id": 999, **(json or {})}}

    async def put(self, path, json=None, params=None):
        self.calls.append(("PUT", path, json, params))
        if path in self.put_errors:
            raise self.put_errors[path]
        return {"value": {"id": 999, **(json or {})}}

    async def delete(self, path):
        self.calls.append(("DELETE", path))
        return {}


class ToolRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_employee_reuses_existing_by_email(self):
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
        self.assertEqual(
            client.calls,
            [("GET", "/employee", {"email": "mia@example.org", "fields": "id,firstName,lastName,email"})],
        )

    async def test_create_customer_reuses_existing_customer_when_flags_match(self):
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

        result = await _execute(
            client,
            "create_customer",
            {
                "name": "Silberberg GmbH",
                "organizationNumber": "871719500",
                "isCustomer": True,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 77)
        self.assertEqual(
            client.calls,
            [("GET", "/customer", {"organizationNumber": "871719500", "fields": "id,name,organizationNumber,isCustomer,isSupplier"})],
        )

    async def test_create_customer_returns_existing_when_flag_upgrade_put_fails(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/customer",
                    (("fields", "id,name,organizationNumber,isCustomer,isSupplier"), ("organizationNumber", "871719500")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 77, "name": "Silberberg GmbH", "isCustomer": True, "isSupplier": False}],
                }
            },
            put_errors={"/customer/77": Exception("422 Validation failed: customerNumber is in use")},
        )

        result = await _execute(
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

        self.assertEqual(result["value"]["id"], 77)
        self.assertIn(
            ("PUT", "/customer/77", {"id": 77, "isCustomer": False, "isSupplier": True}, None),
            client.calls,
        )
        self.assertNotIn(("POST", "/customer", {"name": "Silberberg GmbH", "organizationNumber": "871719500", "isCustomer": False, "isSupplier": True}), client.calls)

    async def test_create_product_reuses_existing_product_number(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/product",
                    (("fields", "id,name,number"), ("productNumber", "PROJ-DESIGN-1450")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 88, "name": "Projet - Design", "number": "PROJ-DESIGN-1450"}],
                }
            }
        )

        result = await _execute(
            client,
            "create_product",
            {
                "name": "Projet - Design",
                "number": "PROJ-DESIGN-1450",
                "priceExcludingVatCurrency": 1450,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 88)
        self.assertEqual(
            client.calls,
            [("GET", "/product", {"productNumber": "PROJ-DESIGN-1450", "fields": "id,name,number"})],
        )

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

    async def test_create_order_sets_ex_vat_mode_flag_and_project(self):
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
            ctx=EntityContext(last_customer_id=321, last_project_id=987, product_ids=[654]),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/order"))
        self.assertEqual(body["customer"], {"id": 321})
        self.assertEqual(body["project"], {"id": 987})
        self.assertEqual(body["orderLines"][0]["product"], {"id": 654})
        self.assertEqual(body["orderLines"][0]["vatType"], {"id": 3})
        self.assertFalse(body["isPrioritizeAmountsIncludingVat"])

    async def test_tripletex_api_call_normalizes_vattype_fields(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/vatType",
                    (("fields", "id,name,percentage"),),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 3, "name": "High rate", "percentage": 25.0}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/ledger/vatType?fields=id,name,rate"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["percentage"], 25.0)
        self.assertEqual(
            client.calls,
            [("GET", "/ledger/vatType", {"fields": "id,name,percentage"})],
        )

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
