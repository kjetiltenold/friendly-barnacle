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

    async def test_create_product_resolves_vat_type_from_percentage(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/vatType",
                    (("fields", "id,number,name,percentage"), ("percentage", "15")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 5, "name": "Utgående mva middels sats", "percentage": 15.0},
                        {"id": 52, "name": "Inngående mva middels sats", "percentage": 15.0},
                    ],
                },
                (
                    "/product",
                    (("fields", "id,name,number"), ("productNumber", "4431")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_product",
            {
                "name": "Galletas de avena",
                "number": "4431",
                "priceExcludingVatCurrency": 41050,
                "vatPercentage": 15,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(
            client.calls,
            [
                ("GET", "/ledger/vatType", {"percentage": "15", "fields": "id,number,name,percentage"}),
                ("GET", "/product", {"productNumber": "4431", "fields": "id,name,number"}),
                ("POST", "/product", {
                    "name": "Galletas de avena",
                    "number": "4431",
                    "priceExcludingVatCurrency": 41050,
                    "vatType": {"id": 5},
                }),
            ],
        )

    async def test_create_product_defaults_dunning_fee_to_zero_vat(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/vatType",
                    (("fields", "id,number,name,percentage"), ("percentage", "0")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 7, "number": 7, "name": "Ingen mva", "percentage": 0.0},
                        {"id": 0, "number": 0, "name": "Outgoing no VAT", "percentage": 0.0},
                    ],
                },
                (
                    "/product",
                    (("fields", "id,name,number"), ("productNumber", "MAHNGEBUEHR-60")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_product",
            {
                "name": "Mahngebuhr",
                "number": "MAHNGEBUEHR-60",
                "priceExcludingVatCurrency": 60,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls[-1], ("POST", "/product", {
            "name": "Mahngebuhr",
            "number": "MAHNGEBUEHR-60",
            "priceExcludingVatCurrency": 60,
            "vatType": {"id": 0},
        }))

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

    async def test_create_employment_details_fallback_resolves_prefixed_occupation_code(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/occupationCode",
                    (("code", "3323"), ("count", 20), ("fields", "id,nameNO,code")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/employment/occupationCode",
                    (("count", 200), ("fields", "id,nameNO,code")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 991, "code": "3323.01", "nameNO": "Kontormedarbeider"},
                        {"id": 992, "code": "4110", "nameNO": "Kontorassistent"},
                    ],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2813401"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_employment_details",
            {
                "employmentId": 2813401,
                "date": "2026-07-06",
                "annualSalary": 500000,
                "percentageOfFullTimeEquivalent": 80,
                "occupationCodeCode": "3323",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertIn(
            ("POST", "/employee/employment/details", {
                "employment": {"id": 2813401},
                "date": "2026-07-06",
                "employmentType": "ORDINARY",
                "remunerationType": "MONTHLY_WAGE",
                "workingHoursScheme": "NOT_SHIFT",
                "occupationCode": {"id": 991},
                "percentageOfFullTimeEquivalent": 80.0,
                "annualSalary": 500000.0,
            }),
            client.calls,
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

    async def test_create_order_normalizes_fee_line_to_zero_vat(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/vatType",
                    (("fields", "id,number,name,percentage"), ("percentage", "0")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 7, "number": 7, "name": "Ingen mva", "percentage": 0.0},
                        {"id": 0, "number": 0, "name": "Outgoing no VAT", "percentage": 0.0},
                    ],
                },
            }
        )

        await _execute(
            client,
            "create_order",
            {
                "customer": {"id": 108297625},
                "orderDate": "2026-03-21",
                "deliveryDate": "2026-03-21",
                "orderLines": [
                    {
                        "product": {"id": 84414648},
                        "description": "Mahngebuhr fur uberfallige Rechnung 3",
                        "count": 1,
                        "unitPriceExcludingVatCurrency": 60,
                        "vatType": {"id": 7},
                    }
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls[-1], ("POST", "/order", {
            "customer": {"id": 108297625},
            "orderDate": "2026-03-21",
            "deliveryDate": "2026-03-21",
            "orderLines": [
                {
                    "product": {"id": 84414648},
                    "description": "Mahngebuhr fur uberfallige Rechnung 3",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": 60,
                    "vatType": {"id": 0},
                }
            ],
            "isPrioritizeAmountsIncludingVat": False,
        }))

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

    async def test_tripletex_api_call_normalizes_invoice_fields(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/invoice",
                    (
                        ("fields", "id,invoiceNumber,invoiceDueDate,amountOutstanding,amount"),
                        ("invoiceDateFrom", "2000-01-01"),
                        ("invoiceDateTo", "2100-01-01"),
                        ("isPaid", "false"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 1, "invoiceNumber": 3, "invoiceDueDate": "2026-03-20", "amountOutstanding": 5000, "amount": 10000}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/invoice?isPaid=false&fields=id,invoiceNumber,dueDate,amountRemainder,amountTotal,amountGross"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["amountOutstanding"], 5000)
        self.assertEqual(result["values"][0]["invoiceDueDate"], "2026-03-20")
        self.assertEqual(
            client.calls,
            [("GET", "/invoice", {"isPaid": "false", "fields": "id,invoiceNumber,invoiceDueDate,amountOutstanding,amount", "invoiceDateFrom": "2000-01-01", "invoiceDateTo": "2100-01-01"})],
        )

    async def test_tripletex_api_call_injects_supplier_invoice_dates(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/supplierInvoice",
                    (
                        ("fields", "*"),
                        ("invoiceDateFrom", "2000-01-01"),
                        ("invoiceDateTo", "2100-01-01"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 1, "invoiceNumber": "SI-1"}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/supplierInvoice?fields=*"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["invoiceNumber"], "SI-1")
        self.assertEqual(
            client.calls,
            [("GET", "/supplierInvoice", {"fields": "*", "invoiceDateFrom": "2000-01-01", "invoiceDateTo": "2100-01-01"})],
        )

    async def test_tripletex_api_call_does_not_inject_invoice_dates_on_payment_type(self):
        client = FakeTripletexClient(
            get_responses={
                ("/invoice/paymentType", ()): {"fullResultSize": 1, "values": [{"id": 1, "description": "Bank"}]},
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/invoice/paymentType"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["id"], 1)
        self.assertEqual(client.calls, [("GET", "/invoice/paymentType", {})])

    async def test_tripletex_api_call_enriches_ledger_account_fields(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (
                        ("fields", "id,number,name,vatType,legalVatTypes,vatLocked,requiresDepartment,isApplicableForSupplierInvoice,isBankAccount"),
                        ("number", "7350"),
                    ),
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
            [("GET", "/ledger/account", {"number": "7350", "fields": "id,number,name,vatType,legalVatTypes,vatLocked,requiresDepartment,isApplicableForSupplierInvoice,isBankAccount"})],
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

    async def test_create_voucher_retries_with_customer_on_receivable_posting(self):
        client = FakeTripletexClient(
            post_errors={
                "/ledger/voucher": [
                    Exception("422 Validation failed: Kunde mangler."),
                    None,
                ]
            }
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Mahngebuhr",
                "postings": [
                    {"account": {"id": 1500}, "amountGross": 60},
                    {"account": {"id": 3400}, "amountGross": -60},
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(last_customer_id=108297625),
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[1][2]["postings"][0]["customer"], {"id": 108297625})

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


    async def test_delete_travel_expense_by_email(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee",
                    (("count", 1), ("email", "charles@example.org"), ("fields", "id")),
                ): {"fullResultSize": 1, "values": [{"id": 42}]},
                (
                    "/travelExpense",
                    (("count", 100), ("employeeId", 42), ("fields", "id,title")),
                ): {"fullResultSize": 1, "values": [{"id": 777, "title": "Client visit"}]},
            }
        )

        await _execute(
            client,
            "delete_travel_expense",
            {"employee_email": "charles@example.org"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls[-1], ("DELETE", "/travelExpense/777"))

    async def test_delete_travel_expense_by_email_and_title(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee",
                    (("count", 1), ("email", "charles@example.org"), ("fields", "id")),
                ): {"fullResultSize": 1, "values": [{"id": 42}]},
                (
                    "/travelExpense",
                    (("count", 100), ("employeeId", 42), ("fields", "id,title")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 777, "title": "Client visit"},
                        {"id": 778, "title": "Conference"},
                    ],
                },
            }
        )

        await _execute(
            client,
            "delete_travel_expense",
            {"employee_email": "charles@example.org", "title": "Conference"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls[-1], ("DELETE", "/travelExpense/778"))

    async def test_delete_travel_expense_requires_title_when_multiple_exist(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee",
                    (("count", 1), ("email", "charles@example.org"), ("fields", "id")),
                ): {"fullResultSize": 1, "values": [{"id": 42}]},
                (
                    "/travelExpense",
                    (("count", 100), ("employeeId", 42), ("fields", "id,title")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 777, "title": "Client visit"},
                        {"id": 778, "title": "Conference"},
                    ],
                },
            }
        )

        with self.assertRaises(ValueError):
            await _execute(
                client,
                "delete_travel_expense",
                {"employee_email": "charles@example.org"},
                endpoint_search=None,
                ctx=EntityContext(),
            )

    async def test_delete_travel_expense_by_id(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "delete_travel_expense",
            {"travel_expense_id": 888},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls, [("DELETE", "/travelExpense/888")])

    async def test_reverse_voucher(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "reverse_voucher",
            {"voucher_id": 555, "date": "2026-03-21"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(
            client.calls,
            [("PUT", "/ledger/voucher/555/:reverse", None, {"date": "2026-03-21"})],
        )

    async def test_create_voucher_rejects_unbalanced_postings(self):
        client = FakeTripletexClient()

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Unbalanced voucher",
                "postings": [
                    {"account": {"id": 100}, "amountGross": 5000},
                    {"account": {"id": 200}, "amountGross": -3000},
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertIn("error", result)
        self.assertIn("do not balance", result["error"])
        # No API call should have been made
        self.assertEqual(client.calls, [])

    async def test_create_voucher_accepts_balanced_postings(self):
        client = FakeTripletexClient()

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Balanced voucher",
                "postings": [
                    {"account": {"id": 100}, "amountGross": 5000},
                    {"account": {"id": 200}, "amountGross": -5000},
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertNotIn("error", result)
        self.assertEqual(client.calls[0][0:2], ("POST", "/ledger/voucher"))

    def test_entity_context_tracks_voucher_id(self):
        ctx = EntityContext()
        ctx.track("create_voucher", {"value": {"id": 12345}})
        self.assertEqual(ctx.last_voucher_id, 12345)

    def test_vat_type_cache_populated_from_lookup(self):
        from app.agent.tools import _track_lookup_context

        ctx = EntityContext()
        _track_lookup_context(ctx, "/ledger/vatType", {
            "values": [
                {"id": 3, "percentage": 25, "name": "Utgående mva høy sats"},
                {"id": 50, "percentage": 25, "name": "Inngående mva høy sats"},
                {"id": 5, "percentage": 15, "name": "Utgående mva middels sats"},
            ],
        })

        self.assertEqual(ctx.last_vat_type_id, 3)
        self.assertIsNotNone(ctx.vat_type_cache)
        self.assertEqual(ctx.vat_type_cache[(25, "outgoing")], 3)
        self.assertEqual(ctx.vat_type_cache[(25, "incoming")], 50)
        self.assertEqual(ctx.vat_type_cache[(15, "outgoing")], 5)

    def test_account_cache_populated_from_lookup(self):
        from app.agent.tools import _track_lookup_context

        ctx = EntityContext()
        _track_lookup_context(ctx, "/ledger/account", {
            "values": [
                {
                    "id": 364015653,
                    "number": 7140,
                    "name": "Reisekostnad",
                    "vatType": {"id": 61},
                    "legalVatTypes": [{"id": 61}, {"id": 62}],
                    "vatLocked": False,
                },
                {
                    "id": 364015350,
                    "number": 1920,
                    "name": "Bank",
                    "isBankAccount": True,
                },
            ],
        })

        self.assertEqual(ctx.last_account_id, 364015653)
        self.assertEqual(ctx.account_cache[364015653]["number"], 7140)
        self.assertTrue(ctx.account_cache[364015350]["isBankAccount"])

    async def test_create_voucher_receipt_uses_account_default_vat_type(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            last_department_id=916856,
            account_cache={
                364015653: {
                    "id": 364015653,
                    "number": 7140,
                    "name": "Reisekostnad",
                    "vatType": {"id": 61},
                    "legalVatTypes": [{"id": 61}, {"id": 62}],
                    "vatLocked": False,
                },
                364015350: {
                    "id": 364015350,
                    "number": 1920,
                    "name": "Bank",
                    "isBankAccount": True,
                },
            },
        )

        await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-04-13",
                "description": "NSB kvittering - Togbillett",
                "postings": [
                    {
                        "account": {"id": 364015653},
                        "amountGross": 14100,
                        "description": "Togbillett",
                        "department": {"id": 916856},
                        "vatType": {"id": 50},
                    },
                    {
                        "account": {"id": 364015350},
                        "amountGross": -14100,
                        "description": "Betalt med bedriftskort",
                    },
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/ledger/voucher"))
        self.assertEqual(body["postings"][0]["vatType"], {"id": 61})

    async def test_create_voucher_removes_locked_vat_type_before_post(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            account_cache={
                362775685: {
                    "id": 362775685,
                    "number": 7350,
                    "name": "Representasjon",
                    "vatType": {"id": 0},
                    "legalVatTypes": [],
                    "vatLocked": True,
                },
                362775433: {
                    "id": 362775433,
                    "number": 1920,
                    "name": "Bank",
                    "isBankAccount": True,
                },
            },
        )

        await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-09",
                "description": "Receipt",
                "postings": [
                    {"account": {"id": 362775685}, "amountGross": 13200, "vatType": {"id": 1}},
                    {"account": {"id": 362775433}, "amountGross": -13200},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/ledger/voucher"))
        self.assertNotIn("vatType", body["postings"][0])


if __name__ == "__main__":
    unittest.main()
