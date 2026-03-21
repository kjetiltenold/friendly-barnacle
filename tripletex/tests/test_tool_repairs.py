import unittest
from unittest.mock import ANY

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
            error = self.post_errors[path]
            if isinstance(error, list):
                current = error.pop(0)
                if current is not None:
                    raise current
            else:
                raise error
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
                    (("email", "mia@example.org"), ("fields", "id,firstName,lastName,email,department")),
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
            [("GET", "/employee", {"email": "mia@example.org", "fields": "id,firstName,lastName,email,department"})],
        )

    async def test_create_employee_uses_context_department(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_employee",
            {
                "firstName": "Lars",
                "lastName": "Strand",
                "email": "lars.strand@example.com",
                "dateOfBirth": "1982-08-04",
                "startDate": "2026-06-24",
            },
            endpoint_search=None,
            ctx=EntityContext(last_department_id=926884),
        )

        self.assertEqual(client.calls[0][0:2], ("GET", "/employee"))
        self.assertEqual(client.calls[1][0:2], ("POST", "/employee"))
        self.assertEqual(client.calls[1][2]["department"], {"id": 926884})

    async def test_create_employee_without_email_defaults_to_no_access_and_normalizes_identity(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_employee",
            {
                "firstName": "Marit",
                "lastName": "Lunde",
                "dateOfBirth": "1982-09-19",
                "nationalIdentityNumber": "190982 12345",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls, [("POST", "/employee", {
            "firstName": "Marit",
            "lastName": "Lunde",
            "dateOfBirth": "1982-09-19",
            "nationalIdentityNumber": "19098212345",
            "userType": "NO_ACCESS",
        })])

    async def test_create_employee_updates_existing_department_when_requested(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee",
                    (("email", "lars.strand@example.com"), ("fields", "id,firstName,lastName,email,department")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 42, "email": "lars.strand@example.com", "department": {"id": 709031}}],
                }
            }
        )

        result = await _execute(
            client,
            "create_employee",
            {
                "firstName": "Lars",
                "lastName": "Strand",
                "email": "lars.strand@example.com",
                "department": {"id": 926884},
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["department"], {"id": 926884})
        self.assertEqual(
            client.calls,
            [
                ("GET", "/employee", {"email": "lars.strand@example.com", "fields": "id,firstName,lastName,email,department"}),
                ("PUT", "/employee/42", {"id": 42, "department": {"id": 926884}}, None),
            ],
        )

    async def test_update_employee_uses_context_department_when_fields_missing(self):
        client = FakeTripletexClient()

        result = await _execute(
            client,
            "update_employee",
            {"employee_id": 42},
            endpoint_search=None,
            ctx=EntityContext(last_department_id=926884),
        )

        self.assertEqual(result["value"]["department"], {"id": 926884})
        self.assertEqual(
            client.calls,
            [("PUT", "/employee/42", {"id": 42, "department": {"id": 926884}}, None)],
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

    async def test_create_project_uses_existing_project_manager_when_missing(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/project",
                    (("count", 50), ("fields", "id,projectManager")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 500, "projectManager": {"id": 901}}],
                }
            }
        )

        await _execute(
            client,
            "create_project",
            {"name": "Internt prosjekt"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/project"))
        self.assertEqual(body["projectManager"], {"id": 901})
        self.assertTrue(body["isInternal"])

    async def test_create_project_retries_with_existing_project_manager_on_validation_error(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/project",
                    (("count", 50), ("fields", "id,projectManager")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 500, "projectManager": {"id": 901}}],
                }
            },
            post_errors={"/project": [Exception("422 Validation failed: projectManager.id invalid"), None]},
        )

        result = await _execute(
            client,
            "create_project",
            {"name": "Internt prosjekt", "projectManager": {"id": 200}},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[0][0:2], ("POST", "/project"))
        self.assertEqual(client.calls[0][2]["projectManager"], {"id": 200})
        self.assertTrue(client.calls[0][2]["isInternal"])
        self.assertEqual(client.calls[0][2]["name"], "Internt prosjekt")
        self.assertEqual(client.calls[0][2]["number"], ANY)
        self.assertEqual(client.calls[0][2]["startDate"], ANY)
        self.assertEqual(client.calls[1], ("GET", "/project", {"fields": "id,projectManager", "count": 50}))
        self.assertEqual(client.calls[2][0:2], ("POST", "/project"))
        self.assertEqual(client.calls[2][2]["projectManager"], {"id": 901})

    async def test_create_department_reuses_existing_by_name(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/department",
                    (("count", 10), ("fields", "id,name"), ("name", "Salg")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 55, "name": "Salg"}],
                }
            }
        )

        result = await _execute(
            client,
            "create_department",
            {"name": "Salg"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 55)
        self.assertEqual(
            client.calls,
            [("GET", "/department", {"name": "Salg", "fields": "id,name", "count": 10})],
        )

    def test_entity_context_tracks_employment_from_employee_create(self):
        ctx = EntityContext()

        ctx.track(
            "create_employee",
            {"value": {"id": 42, "employments": [{"id": 2813136, "startDate": "2026-06-24"}]}},
        )

        self.assertEqual(ctx.last_employee_id, 42)
        self.assertEqual(ctx.last_employment_id, 2813136)

    async def test_create_employment_details_upserts_department_and_standard_time(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/workingHoursScheme",
                    (("count", 1), ("fields", "id,workingHoursScheme,nameNO,code"), ("id", "50")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 50, "workingHoursScheme": "NOT_SHIFT", "nameNO": "Ikke skift", "code": "NS"}],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2813136"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/standardTime",
                    (("count", 100), ("employeeId", 18618852), ("fields", "id,fromDate,hoursPerDay")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        result = await _execute(
            client,
            "create_employment_details",
            {
                "employmentId": 2813136,
                "employeeId": 18618852,
                "fromDate": "2026-06-24",
                "salary": 800000,
                "employmentPercentage": 100,
                "hoursPerDay": 7.5,
                "departmentId": 926884,
                "workingHoursSchemeId": 50,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["employment"], {"id": 2813136})
        self.assertEqual(
            client.calls,
            [
                ("PUT", "/employee/18618852", {"id": 18618852, "department": {"id": 926884}}, None),
                ("GET", "/employee/employment/workingHoursScheme", {"id": "50", "fields": "id,workingHoursScheme,nameNO,code", "count": 1}),
                ("GET", "/employee/employment/details", {"employmentId": "2813136", "fields": "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme", "count": 100}),
                ("POST", "/employee/employment/details", {
                    "employment": {"id": 2813136},
                    "date": "2026-06-24",
                    "employmentType": "ORDINARY",
                    "remunerationType": "MONTHLY_WAGE",
                    "workingHoursScheme": "NOT_SHIFT",
                    "percentageOfFullTimeEquivalent": 100.0,
                    "annualSalary": 800000.0,
                }),
                ("GET", "/employee/standardTime", {"employeeId": 18618852, "fields": "id,fromDate,hoursPerDay", "count": 100}),
                ("POST", "/employee/standardTime", {
                    "employee": {"id": 18618852},
                    "fromDate": "2026-06-24",
                    "hoursPerDay": 7.5,
                }),
            ],
        )

    async def test_create_employment_details_defaults_to_ordinary_and_resolves_occupation_code(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/occupationCode",
                    (("code", "2512"), ("count", 20), ("fields", "id,nameNO,code")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 881, "code": "2512", "nameNO": "Programvareutvikler"}],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2813136"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_employment_details",
            {
                "employmentId": 2813136,
                "date": "2026-07-25",
                "annualSalary": 480000,
                "percentageOfFullTimeEquivalent": 80,
                "stillingskode": "2512",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(
            client.calls,
            [
                ("GET", "/employee/employment/2813136", None),
                ("GET", "/employee/employment/occupationCode", {"code": "2512", "fields": "id,nameNO,code", "count": 20}),
                ("GET", "/employee/employment/details", {"employmentId": "2813136", "fields": "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme", "count": 100}),
                ("POST", "/employee/employment/details", {
                    "employment": {"id": 2813136},
                    "date": "2026-07-25",
                    "employmentType": "ORDINARY",
                    "remunerationType": "MONTHLY_WAGE",
                    "workingHoursScheme": "NOT_SHIFT",
                    "occupationCode": {"id": 881},
                    "percentageOfFullTimeEquivalent": 80.0,
                    "annualSalary": 480000.0,
                }),
            ],
        )

    async def test_tripletex_api_call_normalizes_raw_employment_details_post(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/workingHoursScheme",
                    (("count", 1), ("fields", "id,workingHoursScheme,nameNO,code"), ("id", "50")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 50, "workingHoursScheme": "NOT_SHIFT", "nameNO": "Ikke skift", "code": "NS"}],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2813136"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/standardTime",
                    (("count", 100), ("employeeId", 18618852), ("fields", "id,fromDate,hoursPerDay")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "tripletex_api_call",
            {
                "method": "POST",
                "path": "/employee/employment/details?employmentId=2813136&employeeId=18618852&fromDate=2026-06-24&salary=800000&employmentPercentage=100&hoursPerDay=7.5&departmentId=926884&workingHoursSchemeId=50",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls[0], ("PUT", "/employee/18618852", {"id": 18618852, "department": {"id": 926884}}, None))
        self.assertEqual(client.calls[3][0:2], ("POST", "/employee/employment/details"))
        self.assertEqual(client.calls[3][2]["workingHoursScheme"], "NOT_SHIFT")
        self.assertEqual(client.calls[5][0:2], ("POST", "/employee/standardTime"))

    async def test_create_activity_reuses_existing_by_name(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/activity",
                    (("count", 10), ("fields", "id,name"), ("name", "Analyse")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 77, "name": "Analyse"}],
                }
            }
        )

        result = await _execute(
            client,
            "create_activity",
            {"name": "Analyse"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 77)
        self.assertEqual(
            client.calls,
            [("GET", "/activity", {"name": "Analyse", "fields": "id,name", "count": 10})],
        )

    async def test_create_project_activity_pairs_multiple_projects_and_activities(self):
        client = FakeTripletexClient()
        ctx = EntityContext(project_ids=[101, 102], activity_ids=[201, 202])

        await _execute(
            client,
            "create_project_activity",
            {},
            endpoint_search=None,
            ctx=ctx,
        )
        await _execute(
            client,
            "create_project_activity",
            {},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls,
            [
                ("POST", "/project/projectActivity", {"project": {"id": 101}, "activity": {"id": 201}}),
                ("POST", "/project/projectActivity", {"project": {"id": 102}, "activity": {"id": 202}}),
            ],
        )

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

    async def test_tripletex_api_call_enriches_ledger_account_fields(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (("fields", "id,number,name,vatType,vatLocked,requiresDepartment,isApplicableForSupplierInvoice"), ("number", "7350")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 10, "number": 7350, "name": "Representasjon", "vatLocked": True}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/ledger/account?number=7350&fields=id,number,name"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertTrue(result["values"][0]["vatLocked"])
        self.assertEqual(
            client.calls,
            [("GET", "/ledger/account", {"number": "7350", "fields": "id,number,name,vatType,vatLocked,requiresDepartment,isApplicableForSupplierInvoice"})],
        )

    async def test_tripletex_api_call_blocks_session_endpoints(self):
        client = FakeTripletexClient()

        with self.assertRaises(ValueError):
            await _execute(
                client,
                "tripletex_api_call",
                {"method": "GET", "path": "/token/session/whoAmI"},
                endpoint_search=None,
                ctx=EntityContext(),
            )
        self.assertEqual(client.calls, [])

    async def test_find_top_expense_account_increases_aggregates_postings(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/postingByDate",
                    (("count", 1000), ("dateFrom", "2026-01-01"), ("dateTo", "2026-02-01"), ("from", 0)),
                ): {
                    "fullResultSize": 3,
                    "values": [
                        {"account": {"id": 1, "number": 5000, "name": "Lønn"}, "amount": 1000},
                        {"account": {"id": 2, "number": 7100, "name": "Bilgodtgjørelse"}, "amount": 300},
                        {"account": {"id": 3, "number": 1500, "name": "Kundefordringer"}, "amount": 999},
                    ],
                },
                (
                    "/ledger/postingByDate",
                    (("count", 1000), ("dateFrom", "2026-02-01"), ("dateTo", "2026-03-01"), ("from", 0)),
                ): {
                    "fullResultSize": 3,
                    "values": [
                        {"account": {"id": 1, "number": 5000, "name": "Lønn"}, "amount": 1800},
                        {"account": {"id": 2, "number": 7100, "name": "Bilgodtgjørelse"}, "amount": 900},
                        {"account": {"id": 4, "number": 6500, "name": "Verktøy"}, "amount": 700},
                    ],
                },
            }
        )

        result = await _execute(
            client,
            "find_top_expense_account_increases",
            {
                "period_a_from": "2026-01-01",
                "period_a_to": "2026-02-01",
                "period_b_from": "2026-02-01",
                "period_b_to": "2026-03-01",
                "top_n": 3,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual([item["account"]["number"] for item in result["topAccounts"]], [5000, 6500, 7100])
        self.assertEqual(result["topAccounts"][0]["increase"], 800.0)

    async def test_create_voucher_injects_department_into_positive_posting(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-09",
                "description": "Receipt",
                "postings": [
                    {"account": {"id": 10}, "amountGross": 13200},
                    {"account": {"id": 20}, "amountGross": -13200},
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(last_department_id=44),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/ledger/voucher"))
        self.assertEqual(body["postings"][0]["department"], {"id": 44})
        self.assertNotIn("department", body["postings"][1])

    async def test_create_voucher_retries_without_locked_vattype(self):
        client = FakeTripletexClient(
            post_errors={
                "/ledger/voucher": [
                    Exception("422 Validation failed: Kontoen er låst til mva-kode 0: Ingen avgiftsbehandling."),
                    None,
                ]
            }
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-09",
                "description": "Receipt",
                "postings": [
                    {"account": {"id": 10}, "amountGross": 13200, "vatType": {"id": 1}},
                    {"account": {"id": 20}, "amountGross": -13200},
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[0][0:2], ("POST", "/ledger/voucher"))
        self.assertEqual(client.calls[1][0:2], ("POST", "/ledger/voucher"))
        self.assertIn("vatType", client.calls[0][2]["postings"][0])
        self.assertNotIn("vatType", client.calls[1][2]["postings"][0])

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
