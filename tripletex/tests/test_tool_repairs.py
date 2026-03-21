import datetime
import unittest
import httpx
from unittest.mock import ANY

from app.agent.tools import EntityContext, _execute


class FakeTripletexClient:
    def __init__(self, get_responses=None, post_errors=None, put_errors=None, post_responses=None, put_responses=None):
        self.get_responses = get_responses or {}
        self.post_errors = post_errors or {}
        self.put_errors = put_errors or {}
        self.post_responses = post_responses or {}
        self.put_responses = put_responses or {}
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
        if path in self.post_responses:
            return self.post_responses[path]
        return {"value": {"id": 999, **(json or {})}}

    async def put(self, path, json=None, params=None):
        self.calls.append(("PUT", path, json, params))
        if path in self.put_errors:
            raise self.put_errors[path]
        if path in self.put_responses:
            return self.put_responses[path]
        return {"value": {"id": 999, **(json or {})}}

    async def delete(self, path):
        self.calls.append(("DELETE", path))
        return {}


class ToolRepairTests(unittest.IsolatedAsyncioTestCase):
    def _http_status_error(self, path: str, message: str, status_code: int = 422):
        request = httpx.Request("POST", f"https://example.invalid{path}")
        response = httpx.Response(status_code, request=request, text=message)
        return httpx.HTTPStatusError(message, request=request, response=response)

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

    async def test_create_employee_strips_placeholder_email_for_contract_task_without_prompt_email(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_employee",
            {
                "firstName": "Andres",
                "lastName": "Perez",
                "email": "andres.perez@example.org",
                "userType": "STANDARD",
                "dateOfBirth": "1980-05-20",
                "nationalIdentityNumber": "20058004635",
                "startDate": "2026-07-11",
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_department_id=943935,
                prompt_text=(
                    "Has recibido un contrato de trabajo para un nuevo empleado. "
                    "Crea el empleado con numero de identidad, fecha de nacimiento, departamento, "
                    "codigo de ocupacion, salario, porcentaje de empleo y fecha de inicio."
                ),
            ),
        )

        self.assertEqual(client.calls, [("POST", "/employee", {
            "firstName": "Andres",
            "lastName": "Perez",
            "userType": "NO_ACCESS",
            "dateOfBirth": "1980-05-20",
            "nationalIdentityNumber": "20058004635",
            "department": {"id": 943935},
            "employments": [{"startDate": "2026-07-11"}],
        })])

    async def test_create_employee_defaults_to_no_access_and_skips_prefetched_department_for_generic_new_employee_prompt(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_employee",
            {
                "firstName": "Lucia",
                "lastName": "Hernandez",
                "email": "lucia.hernandez@example.org",
                "dateOfBirth": "1986-03-14",
                "userType": "STANDARD",
                "startDate": "2026-01-17",
            },
            endpoint_search=None,
            ctx=EntityContext(
                prompt_text=(
                    "Tenemos un nuevo empleado llamado Lucia Hernandez, nacido el 14. March 1986. "
                    "Creelo como empleado con el correo lucia.hernandez@example.org y fecha de inicio 17. January 2026."
                ),
                last_department_id=736856,
                last_department_id_prefetched=True,
            ),
        )

        self.assertEqual(client.calls, [
            ("GET", "/employee", {"email": "lucia.hernandez@example.org", "fields": "id,firstName,lastName,email,department"}),
            ("POST", "/employee", {
                "firstName": "Lucia",
                "lastName": "Hernandez",
                "email": "lucia.hernandez@example.org",
                "dateOfBirth": "1986-03-14",
                "userType": "NO_ACCESS",
                "employments": [{"startDate": "2026-01-17"}],
            }),
        ])

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

    async def test_create_customer_corrects_flags_after_create(self):
        client = FakeTripletexClient(
            post_responses={
                "/customer": {
                    "value": {
                        "id": 88,
                        "name": "Bergvik AS",
                        "organizationNumber": "919398051",
                        "isCustomer": True,
                        "isSupplier": True,
                    }
                }
            },
            put_responses={
                "/customer/88": {
                    "value": {
                        "id": 88,
                        "name": "Bergvik AS",
                        "organizationNumber": "919398051",
                        "isCustomer": False,
                        "isSupplier": True,
                    }
                }
            },
        )

        result = await _execute(
            client,
            "create_customer",
            {
                "name": "Bergvik AS",
                "organizationNumber": "919398051",
                "isCustomer": False,
                "isSupplier": True,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 88)
        self.assertFalse(result["value"]["isCustomer"])
        self.assertTrue(result["value"]["isSupplier"])
        self.assertIn(
            ("PUT", "/customer/88", {"id": 88, "isCustomer": False}, None),
            client.calls,
        )
        self.assertNotIn(("POST", "/customer", {"name": "Silberberg GmbH", "organizationNumber": "871719500", "isCustomer": False, "isSupplier": True}), client.calls)

    async def test_create_customer_corrects_flags_after_create_when_put_response_is_stale(self):
        client = FakeTripletexClient(
            post_responses={
                "/customer": {
                    "value": {
                        "id": 108341268,
                        "name": "Bruckentor GmbH",
                        "organizationNumber": "981448294",
                        "isCustomer": True,
                        "isSupplier": True,
                    }
                }
            },
            put_responses={
                "/customer/108341268": {
                    "value": {
                        "id": 108341268,
                        "name": "Bruckentor GmbH",
                        "organizationNumber": "981448294",
                        "isCustomer": True,
                        "isSupplier": True,
                    }
                }
            },
        )

        result = await _execute(
            client,
            "create_customer",
            {
                "name": "Bruckentor GmbH",
                "organizationNumber": "981448294",
                "isCustomer": False,
                "isSupplier": True,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 108341268)
        self.assertFalse(result["value"]["isCustomer"])
        self.assertTrue(result["value"]["isSupplier"])

    async def test_create_customer_retries_lookup_and_does_not_create_new_customer_for_invoice_payment_task(self):
        class RetryLookupClient(FakeTripletexClient):
            def __init__(self):
                super().__init__()
                self.customer_lookup_attempts = 0

            async def get(self, path, params=None):
                self.calls.append(("GET", path, params))
                if path == "/customer":
                    self.customer_lookup_attempts += 1
                    if self.customer_lookup_attempts == 1:
                        raise RuntimeError("500 Internal Server Error: <!DOCTYPE html><title>Feilsituasjon - Tripletex</title>")
                    return {
                        "fullResultSize": 1,
                        "values": [
                            {
                                "id": 108393698,
                                "name": "Rio Azul Lda",
                                "organizationNumber": "932217643",
                                "isCustomer": True,
                                "isSupplier": False,
                            }
                        ],
                    }
                return await super().get(path, params)

        client = RetryLookupClient()

        result = await _execute(
            client,
            "create_customer",
            {
                "name": "Rio Azul Lda",
                "organizationNumber": "932217643",
                "isCustomer": True,
            },
            endpoint_search=None,
            ctx=EntityContext(
                prompt_text='O cliente Rio Azul Lda (org. nº 932217643) tem uma fatura pendente de 44100 NOK sem IVA por "Sessão de formação". Registe o pagamento total desta fatura.'
            ),
        )

        self.assertEqual(result["value"]["id"], 108393698)
        self.assertEqual(
            client.calls,
            [
                ("GET", "/customer", {"organizationNumber": "932217643", "fields": "id,name,organizationNumber,isCustomer,isSupplier"}),
                ("GET", "/customer", {"organizationNumber": "932217643", "fields": "id,name,organizationNumber,isCustomer,isSupplier"}),
            ],
        )

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

    async def test_create_product_corrects_zero_vat_fee_product_after_create_when_put_response_is_stale(self):
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
            },
            post_responses={
                "/product": {
                    "value": {
                        "id": 84415595,
                        "name": "Mahngebuhr",
                        "number": "MAHNGEBUEHR-60",
                        "vatType": {"id": 3},
                    }
                }
            },
            put_responses={
                "/product/84415595": {
                    "value": {
                        "id": 84415595,
                        "name": "Mahngebuhr",
                        "number": "MAHNGEBUEHR-60",
                        "vatType": {"id": 3},
                    }
                }
            },
        )

        result = await _execute(
            client,
            "create_product",
            {
                "name": "Mahngebuhr",
                "number": "MAHNGEBUEHR-60",
                "priceExcludingVatCurrency": 60,
                "vatPercentage": 0,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["vatType"]["id"], 0)
        self.assertIn(
            ("PUT", "/product/84415595", {"id": 84415595, "vatType": {"id": 0}}, None),
            client.calls,
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

    async def test_search_entity_accepts_top_level_filter_fields(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/department",
                    (("fields", "id,name"), ("name", "Produksjon")),
                ): {"fullResultSize": 1, "values": [{"id": 928225, "name": "Produksjon"}]}
            }
        )
        ctx = EntityContext()

        result = await _execute(
            client,
            "search_entity",
            {"entity_type": "department", "name": "Produksjon", "fields": "id,name"},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["values"][0]["id"], 928225)
        self.assertEqual(ctx.last_department_id, 928225)
        self.assertEqual(
            client.calls,
            [("GET", "/department", {"name": "Produksjon", "fields": "id,name"})],
        )

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

    async def test_create_project_prefers_sales_customer_and_first_employee(self):
        client = FakeTripletexClient()
        ctx = EntityContext()
        ctx.track(
            "create_customer",
            {"value": {"id": 108293203, "name": "Brattli AS", "isCustomer": True, "isSupplier": False}},
            {"isCustomer": True},
        )
        ctx.track(
            "create_customer",
            {"value": {"id": 108336989, "name": "Lysgard AS", "isCustomer": True, "isSupplier": True}},
            {"isCustomer": False, "isSupplier": True},
        )
        ctx.track("create_employee", {"value": {"id": 18596035}}, {"email": "hilde.degard@example.org"})
        ctx.track("create_employee", {"value": {"id": 18596063}}, {"email": "lars.johansen@example.org"})

        await _execute(
            client,
            "create_project",
            {"name": "Dataplattform Brattli", "number": "DPB-2026-001", "startDate": "2026-03-21"},
            endpoint_search=None,
            ctx=ctx,
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/project"))
        self.assertEqual(body["customer"], {"id": 108293203})
        self.assertEqual(body["projectManager"], {"id": 18596035})

    async def test_create_project_normalizes_fixed_price_alias(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_project",
            {
                "name": "Migracao para nuvem",
                "number": "PRJ-20260321-001",
                "startDate": "2026-03-21",
                "isFixedPrice": True,
                "fixedPrice": 362300,
            },
            endpoint_search=None,
            ctx=EntityContext(last_customer_id=100, last_employee_id=200),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/project"))
        self.assertEqual(body["customer"], {"id": 100})
        self.assertEqual(body["projectManager"], {"id": 200})
        self.assertTrue(body["isFixedPrice"])
        self.assertEqual(body["fixedprice"], 362300)
        self.assertNotIn("fixedPrice", body)

    async def test_create_project_infers_fixed_price_from_portuguese_prompt(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_project",
            {
                "name": "Melhoria de infraestrutura",
                "number": "PRJ-20260321-050",
                "startDate": "2026-03-21",
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_customer_id=100,
                last_sales_customer_id=100,
                last_employee_id=200,
                employee_ids=[200],
                prompt_text=(
                    "Defina um preço fixo de 228150 NOK no projeto "
                    "\"Melhoria de infraestrutura\" para Horizonte Lda."
                ),
            ),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/project"))
        self.assertTrue(body["isFixedPrice"])
        self.assertEqual(body["fixedprice"], 228150)

    async def test_create_project_does_not_force_fixed_price_when_prompt_only_says_budget(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_project",
            {
                "name": "Portail Numerique Etoile",
                "number": "PNE-2026-001",
                "startDate": "2026-03-21",
                "isFixedPrice": True,
                "fixedprice": 383650,
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_customer_id=108349514,
                last_sales_customer_id=108349514,
                last_employee_id=18626106,
                employee_ids=[18626106],
                prompt_text="Le projet a un budget de 383650 NOK et doit etre facture apres enregistrement du temps.",
            ),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/project"))
        self.assertEqual(body["customer"], {"id": 108349514})
        self.assertEqual(body["projectManager"], {"id": 18626106})
        self.assertNotIn("isFixedPrice", body)
        self.assertNotIn("fixedprice", body)

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

    def test_entity_context_tracks_linked_activity_from_project_activity_create(self):
        ctx = EntityContext(last_activity_id=5173580, last_project_id=402000662)

        ctx.track(
            "create_project_activity",
            {"value": {"id": 19801646, "project": {"id": 402000662}, "activity": {"id": 5877567}}},
        )

        self.assertEqual(ctx.last_project_id, 402000662)
        self.assertEqual(ctx.last_activity_id, 5877567)

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

    async def test_create_standard_time_noops_when_existing_value_matches(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/standardTime",
                    (("count", 100), ("employeeId", 18622953), ("fields", "id,fromDate,hoursPerDay")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 43748, "fromDate": "2026-06-24", "hoursPerDay": 6.0}],
                }
            }
        )

        result = await _execute(
            client,
            "create_standard_time",
            {
                "employeeId": 18622953,
                "fromDate": "2026-06-24",
                "hoursPerDay": 6,
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 43748)
        self.assertEqual(
            client.calls,
            [
                ("GET", "/employee/standardTime", {"employeeId": 18622953, "fields": "id,fromDate,hoursPerDay", "count": 100}),
            ],
        )

    async def test_create_employment_details_auto_creates_missing_employment_from_employee(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment",
                    (("count", 20), ("employeeId", 18610084), ("fields", "id,startDate,endDate")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/18610084",
                    (("fields", "id,firstName,lastName,email,dateOfBirth,department"),),
                ): {"value": {"id": 18610084, "dateOfBirth": "1990-05-01"}},
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2814181"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
            },
            post_responses={
                "/employee/employment": {"value": {"id": 2814181, "employee": {"id": 18610084}, "startDate": "2026-03-21"}},
            },
        )

        result = await _execute(
            client,
            "create_employment_details",
            {
                "employeeId": 18610084,
                "date": "2026-03-21",
                "annualSalary": 516600,
                "percentageOfFullTimeEquivalent": 100,
                "employmentType": "ORDINARY",
                "workingHoursScheme": "NOT_SHIFT",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["employment"], {"id": 2814181})
        self.assertEqual(
            client.calls,
            [
                ("GET", "/employee/employment", {"employeeId": 18610084, "fields": "id,startDate,endDate", "count": 20}),
                ("GET", "/employee/18610084", {"fields": "id,firstName,lastName,email,dateOfBirth,department"}),
                ("POST", "/employee/employment", {"employee": {"id": 18610084, "dateOfBirth": "1990-05-01"}, "startDate": "2026-03-21"}),
                ("GET", "/employee/employment/details", {"employmentId": "2814181", "fields": "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme", "count": 100}),
                ("POST", "/employee/employment/details", {
                    "employment": {"id": 2814181},
                    "date": "2026-03-21",
                    "employmentType": "ORDINARY",
                    "remunerationType": "MONTHLY_WAGE",
                    "workingHoursScheme": "NOT_SHIFT",
                    "percentageOfFullTimeEquivalent": 100.0,
                    "annualSalary": 516600.0,
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

    def test_entity_context_tracks_employment_details_occupation_code_presence(self):
        ctx = EntityContext()

        ctx.track(
            "create_employment_details",
            {
                "value": {
                    "id": 3729000,
                    "employment": {"id": 2829000},
                    "occupationCode": {"id": 991},
                }
            },
            {
                "employmentId": 2829000,
                "date": "2026-06-06",
                "occupationCodeCode": "3323",
            },
        )

        self.assertEqual(ctx.last_employment_details_id, 3729000)
        self.assertEqual(ctx.last_employment_id, 2829000)
        self.assertTrue(ctx.last_employment_details_had_occupation_code)

    async def test_create_employment_details_fallback_resolves_prefixed_occupation_code(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/occupationCode",
                    (("code", "3323"), ("count", 20), ("fields", "id,nameNO,code")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/employment/occupationCode",
                    (("count", 1000), ("fields", "id,nameNO,code")),
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

    async def test_create_employment_details_fallback_resolves_prefixed_occupation_code_beyond_first_200(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/occupationCode",
                    (("code", "3323"), ("count", 20), ("fields", "id,nameNO,code")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 881, "code": "3320", "nameNO": "Lagerarbeider"},
                        {"id": 882, "code": "3321", "nameNO": "Ordrebehandler"},
                    ],
                },
                (
                    "/employee/employment/occupationCode",
                    (("count", 1000), ("fields", "id,nameNO,code")),
                ): {
                    "fullResultSize": 201,
                    "values": [
                        *[
                            {"id": 1000 + i, "code": f"41{i:02d}", "nameNO": f"Yrke {i}"}
                            for i in range(200)
                        ],
                        {"id": 1991, "code": "3323.01", "nameNO": "Kontormedarbeider"},
                    ],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2813402"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_employment_details",
            {
                "employmentId": 2813402,
                "date": "2026-07-07",
                "annualSalary": 500000,
                "percentageOfFullTimeEquivalent": 100,
                "occupationCodeCode": "3323",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertIn(
            ("POST", "/employee/employment/details", {
                "employment": {"id": 2813402},
                "date": "2026-07-07",
                "employmentType": "ORDINARY",
                "remunerationType": "MONTHLY_WAGE",
                "workingHoursScheme": "NOT_SHIFT",
                "occupationCode": {"id": 1991},
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 500000.0,
            }),
            client.calls,
        )

    async def test_create_employment_details_fallback_resolves_transliterated_occupation_name(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/occupationCode",
                    (("count", 20), ("fields", "id,nameNO,code"), ("nameNO", "HR-radgiver")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/employment/occupationCode",
                    (("count", 500), ("fields", "id,nameNO,code")),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {"id": 4679, "code": "1234", "nameNO": "HR-rådgiver"},
                        {"id": 4680, "code": "1235", "nameNO": "Regnskapsmedarbeider"},
                    ],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2815260"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_employment_details",
            {
                "employmentId": 2815260,
                "date": "2026-04-12",
                "annualSalary": 560000,
                "percentageOfFullTimeEquivalent": 100,
                "employmentType": "ORDINARY",
                "employmentForm": "PERMANENT",
                "workingHoursScheme": "NOT_SHIFT",
                "occupationCodeName": "HR-radgiver",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertIn(
            ("POST", "/employee/employment/details", {
                "employment": {"id": 2815260},
                "date": "2026-04-12",
                "employmentType": "ORDINARY",
                "employmentForm": "PERMANENT",
                "remunerationType": "MONTHLY_WAGE",
                "workingHoursScheme": "NOT_SHIFT",
                "occupationCode": {"id": 4679},
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 560000.0,
            }),
            client.calls,
        )

    async def test_create_employment_details_uses_fallback_scan_when_direct_name_search_returns_non_exact_match(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment/occupationCode",
                    (("count", 20), ("fields", "id,nameNO,code"), ("nameNO", "HR-rådgiver")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 4688, "code": "9999", "nameNO": "HR-medarbeider"}],
                },
                (
                    "/employee/employment/occupationCode",
                    (("count", 500), ("fields", "id,nameNO,code")),
                ): {
                    "fullResultSize": 3,
                    "values": [
                        {"id": 4679, "code": "1234", "nameNO": "HR-rådgiver"},
                        {"id": 4680, "code": "1235", "nameNO": "Regnskapsmedarbeider"},
                        {"id": 4688, "code": "9999", "nameNO": "HR-medarbeider"},
                    ],
                },
                (
                    "/employee/employment/details",
                    (("count", 100), ("employmentId", "2815261"), ("fields", "id,date,annualSalary,percentageOfFullTimeEquivalent,workingHoursScheme")),
                ): {"fullResultSize": 0, "values": []},
            }
        )

        await _execute(
            client,
            "create_employment_details",
            {
                "employmentId": 2815261,
                "date": "2026-10-21",
                "annualSalary": 650000,
                "percentageOfFullTimeEquivalent": 100,
                "employmentType": "ORDINARY",
                "employmentForm": "PERMANENT",
                "workingHoursScheme": "NOT_SHIFT",
                "occupationCodeName": "HR-rådgiver",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertIn(
            ("POST", "/employee/employment/details", {
                "employment": {"id": 2815261},
                "date": "2026-10-21",
                "employmentType": "ORDINARY",
                "employmentForm": "PERMANENT",
                "remunerationType": "MONTHLY_WAGE",
                "workingHoursScheme": "NOT_SHIFT",
                "occupationCode": {"id": 4679},
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 650000.0,
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

    async def test_create_project_activity_injects_budget_fee_from_budget_prompt(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            last_project_id=402008098,
            last_activity_id=5892874,
            prompt_text="Prosjektet har budsjett 361050 kr og skal fakturerast etter timar.",
        )

        await _execute(
            client,
            "create_project_activity",
            {},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls[-1],
            (
                "POST",
                "/project/projectActivity",
                {
                    "project": {"id": 402008098},
                    "activity": {"id": 5892874},
                    "budgetFeeCurrency": 361050.0,
                },
            ),
        )

    async def test_create_timesheet_entry_uses_linked_project_activity_context(self):
        client = FakeTripletexClient()
        ctx = EntityContext(last_employee_id=18609430, last_project_id=402000662, last_activity_id=5173580)
        ctx.track(
            "create_project_activity",
            {"value": {"id": 19801646, "project": {"id": 402000662}, "activity": {"id": 5877567}}},
        )

        await _execute(
            client,
            "create_timesheet_entry",
            {"date": "2026-03-21", "hours": 8},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls[-1],
            ("POST", "/timesheet/entry", {
                "date": "2026-03-21",
                "hours": 8,
                "employee": {"id": 18609430},
                "project": {"id": 402000662},
                "activity": {"id": 5877567},
            }),
        )

    async def test_create_timesheet_entry_shifts_pre_start_dates_forward(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            last_employee_id=18596063,
            last_project_id=402000777,
            last_activity_id=5136255,
            project_start_dates={402000777: "2026-03-21"},
            timesheet_hours_by_day={(18596063, 402000777, 5136255, "2026-03-21"): 24.0},
        )

        result = await _execute(
            client,
            "create_timesheet_entry",
            {"date": "2026-03-20", "hours": 24},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls,
            [
                ("POST", "/timesheet/entry", {
                    "date": "2026-03-23",
                    "hours": 8.0,
                    "employee": {"id": 18596063},
                    "project": {"id": 402000777},
                    "activity": {"id": 5136255},
                }),
                ("POST", "/timesheet/entry", {
                    "date": "2026-03-24",
                    "hours": 8.0,
                    "employee": {"id": 18596063},
                    "project": {"id": 402000777},
                    "activity": {"id": 5136255},
                }),
                ("POST", "/timesheet/entry", {
                    "date": "2026-03-25",
                    "hours": 8.0,
                    "employee": {"id": 18596063},
                    "project": {"id": 402000777},
                    "activity": {"id": 5136255},
                }),
            ],
        )
        self.assertEqual(len(result["values"]), 3)

    async def test_create_timesheet_entry_resolves_employee_from_prompt_hours(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Gjennomfør hele prosjektsyklusen for 'Dataplattform Tindra'. "
                "Registrer timer: Astrid Hansen (prosjektleder, astrid.hansen@example.org) 42 timer "
                "og Silje Berg (konsulent, silje.berg@example.org) 145 timer."
            ),
            last_project_id=402000662,
            last_activity_id=5877567,
        )
        ctx.track(
            "create_employee",
            {"value": {"id": 101, "firstName": "Astrid", "lastName": "Hansen", "email": "astrid.hansen@example.org"}},
        )
        ctx.track(
            "create_employee",
            {"value": {"id": 102, "firstName": "Silje", "lastName": "Berg", "email": "silje.berg@example.org"}},
        )
        ctx.linked_project_activity_pairs.add((402000662, 5877567))

        result = await _execute(
            client,
            "create_timesheet_entry",
            {"project": {"id": 402000662}, "activity": {"id": 5877567}, "date": "2026-03-21", "hours": 42},
            endpoint_search=None,
            ctx=ctx,
        )

        timesheet_calls = [call for call in client.calls if call[0:2] == ("POST", "/timesheet/entry")]
        self.assertEqual(len(result["values"]), 6)
        self.assertTrue(timesheet_calls)
        self.assertTrue(all(call[2]["employee"] == {"id": 101} for call in timesheet_calls))

    async def test_create_timesheet_entry_auto_links_project_activity_before_post(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text="Prosjektet har budsjett 432000 kr og skal fakturerast etter timar.",
            last_employee_id=18609430,
            last_project_id=402000662,
            last_activity_id=5877567,
        )

        await _execute(
            client,
            "create_timesheet_entry",
            {"date": "2026-03-24", "hours": 8},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls[0],
            (
                "POST",
                "/project/projectActivity",
                {
                    "project": {"id": 402000662},
                    "activity": {"id": 5877567},
                    "budgetFeeCurrency": 432000.0,
                },
            ),
        )
        self.assertEqual(client.calls[1][0:2], ("POST", "/timesheet/entry"))
        self.assertIn((402000662, 5877567), ctx.linked_project_activity_pairs)

    async def test_create_timesheet_entry_splits_large_hours_into_working_days(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            last_employee_id=18626106,
            last_project_id=402006456,
            last_activity_id=5889922,
        )

        result = await _execute(
            client,
            "create_timesheet_entry",
            {"date": "2026-03-21", "hours": 24},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls,
            [
                ("POST", "/timesheet/entry", {
                    "date": "2026-03-23",
                    "hours": 8.0,
                    "employee": {"id": 18626106},
                    "project": {"id": 402006456},
                    "activity": {"id": 5889922},
                }),
                ("POST", "/timesheet/entry", {
                    "date": "2026-03-24",
                    "hours": 8.0,
                    "employee": {"id": 18626106},
                    "project": {"id": 402006456},
                    "activity": {"id": 5889922},
                }),
                ("POST", "/timesheet/entry", {
                    "date": "2026-03-25",
                    "hours": 8.0,
                    "employee": {"id": 18626106},
                    "project": {"id": 402006456},
                    "activity": {"id": 5889922},
                }),
            ],
        )
        self.assertEqual(len(result["values"]), 3)

    async def test_update_project_hourly_rate_derives_fixed_rate_from_budget_and_total_hours(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Gjennomfør hele prosjektsyklusen for 'Dataplattform Tindra'. "
                "Prosjektet har budsjett 432000 kr. "
                "Registrer timer: Astrid Hansen (prosjektleder, astrid.hansen@example.org) 42 timer "
                "og Silje Berg (konsulent, silje.berg@example.org) 145 timer."
            ),
            last_project_id=402000662,
            last_hourly_rate_id=73,
        )

        await _execute(
            client,
            "update_project_hourly_rate",
            {},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls[-1],
            (
                "PUT",
                "/project/hourlyRates/73",
                {
                    "id": 73,
                    "project": {"id": 402000662},
                    "fixedRate": 2310.16,
                    "startDate": datetime.date.today().isoformat(),
                    "hourlyRateModel": "TYPE_FIXED_HOURLY_RATE",
                    "showInProjectOrder": True,
                },
                None,
            ),
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

    async def test_create_order_normalizes_fixed_price_milestone_amount_from_prompt(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_order",
            {
                "orderLines": [
                    {
                        "description": "Pagamento por etapa",
                        "count": 1,
                        "unitPriceExcludingVatCurrency": 228150,
                    }
                ]
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_customer_id=100,
                last_sales_customer_id=100,
                last_project_id=987,
                product_ids=[654],
                prompt_text=(
                    "Defina um preço fixo de 228150 NOK no projeto \"Melhoria de infraestrutura\". "
                    "Fature ao cliente 50 % do preço fixo como pagamento por etapa."
                ),
            ),
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/order"))
        self.assertEqual(body["customer"], {"id": 100})
        self.assertEqual(body["project"], {"id": 987})
        self.assertEqual(body["orderLines"][0]["product"], {"id": 654})
        self.assertEqual(body["orderLines"][0]["unitPriceExcludingVatCurrency"], 114075.0)
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

    async def test_create_order_normalizes_portuguese_fee_line_to_zero_vat(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/vatType",
                    (("fields", "id,number,name,percentage"), ("percentage", "0")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 0, "number": 0, "name": "Outgoing no VAT", "percentage": 0.0}],
                },
            }
        )

        await _execute(
            client,
            "create_order",
            {
                "customer": {"id": 108348803},
                "orderDate": "2026-03-21",
                "deliveryDate": "2026-03-21",
                "orderLines": [
                    {
                        "product": {"id": 84415532},
                        "description": "Taxa de lembrete",
                        "count": 1,
                        "unitPriceExcludingVatCurrency": 35,
                    }
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(client.calls[-1][2]["orderLines"][0]["vatType"], {"id": 0})

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
            {"method": "GET", "path": "/ledger/vatType?fields=id,name,rate,direction"},
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

    async def test_tripletex_api_call_normalizes_invoice_amount_due_and_tracks_customer_context(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/invoice",
                    (
                        ("fields", "id,invoiceNumber,invoiceDate,invoiceDueDate,amountOutstanding,customer"),
                        ("invoiceDateFrom", "2000-01-01"),
                        ("invoiceDateTo", "2026-03-21"),
                        ("isPaid", "false"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [
                        {
                            "id": 2147591804,
                            "invoiceNumber": 2,
                            "invoiceDate": "2026-02-01",
                            "invoiceDueDate": "2026-03-01",
                            "amountOutstanding": 5000,
                            "customer": {"id": 108348803},
                        }
                    ],
                }
            }
        )
        ctx = EntityContext()

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/invoice?invoiceDateTo=2026-03-21&isPaid=false&fields=id,invoiceNumber,invoiceDate,dueDate,amountDue,customer"},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["values"][0]["amountOutstanding"], 5000)
        self.assertEqual(ctx.last_invoice_id, 2147591804)
        self.assertEqual(ctx.last_customer_id, 108348803)
        self.assertEqual(ctx.last_sales_customer_id, 108348803)
        self.assertEqual(
            client.calls,
            [("GET", "/invoice", {"invoiceDateTo": "2026-03-21", "isPaid": "false", "fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,amountOutstanding,customer", "invoiceDateFrom": "2000-01-01"})],
        )

    async def test_tripletex_api_call_drops_invalid_invoice_fields_and_normalizes_sorting(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/invoice",
                    (
                        ("fields", "id,invoiceNumber,invoiceDate,invoiceDueDate,customer,amount,amountOutstanding"),
                        ("invoiceDateFrom", "2025-01-01"),
                        ("invoiceDateTo", "2026-03-22"),
                        ("sorting", "-invoiceDueDate"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [
                        {
                            "id": 2147592359,
                            "invoiceNumber": 4,
                            "invoiceDate": "2026-02-01",
                            "invoiceDueDate": "2026-03-10",
                            "customer": {"id": 108349797},
                            "amount": 12000,
                            "amountOutstanding": 7000,
                        }
                    ],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {
                "method": "GET",
                "path": "/invoice?invoiceDateFrom=2025-01-01&invoiceDateTo=2026-03-22&fields=id,invoiceNumber,invoiceDate,dueDate,customer,amountGross,amountRemainder,isPaid,order&sorting=-dueDate",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["amountOutstanding"], 7000)
        self.assertEqual(
            client.calls,
            [
                (
                    "GET",
                    "/invoice",
                    {
                        "invoiceDateFrom": "2025-01-01",
                        "invoiceDateTo": "2026-03-22",
                        "fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,customer,amount,amountOutstanding",
                        "sorting": "-invoiceDueDate",
                    },
                )
            ],
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

    async def test_tripletex_api_call_injects_invoice_date_for_order_invoice_action(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (
                        ("count", 5),
                        ("fields", "id,number,name,isBankAccount,bankAccountNumber"),
                        ("isBankAccount", "true"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 1920, "number": 1920, "name": "Bank", "isBankAccount": True}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "PUT", "path": "/order/402003015/:invoice"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(
            client.calls,
            [
                (
                    "GET",
                    "/ledger/account",
                    {
                        "fields": "id,number,name,isBankAccount,bankAccountNumber",
                        "isBankAccount": "true",
                        "count": 5,
                    },
                ),
                (
                    "PUT",
                    "/ledger/account/1920",
                    {
                        "id": 1920,
                        "number": 1920,
                        "name": "Bank",
                        "isBankAccount": True,
                        "bankAccountNumber": "12345678903",
                    },
                    None,
                ),
                (
                    "PUT",
                    "/order/402003015/:invoice",
                    None,
                    {"invoiceDate": datetime.date.today().isoformat()},
                )
            ],
        )

    async def test_tripletex_api_call_normalizes_ledger_posting_month_end_date_to_exclusive(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/posting",
                    (
                        ("dateFrom", "2026-03-01"),
                        ("dateTo", "2026-04-01"),
                        ("fields", "amount"),
                    ),
                ): {
                    "fullResultSize": 2,
                    "values": [{"amount": 3500}, {"amount": -3500}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/ledger/posting?dateFrom=2026-03-01&dateTo=2026-03-31&fields=amount"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["fullResultSize"], 2)
        self.assertEqual(
            client.calls,
            [("GET", "/ledger/posting", {"dateFrom": "2026-03-01", "dateTo": "2026-04-01", "fields": "amount"})],
        )

    async def test_tripletex_api_call_normalizes_ledger_posting_fields_and_account_number(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/posting",
                    (
                        ("accountNumberFrom", "1700"),
                        ("accountNumberTo", "1700"),
                        ("dateFrom", "2025-01-01"),
                        ("dateTo", "2026-01-01"),
                        ("fields", "amountGross,account,date"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"amountGross": 61300, "date": "2025-12-31"}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/ledger/posting?dateFrom=2025-01-01&dateTo=2026-01-01&accountNumber=1700&fields=amountGross,account,accountingDate"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["amountGross"], 61300)
        self.assertEqual(
            client.calls,
            [("GET", "/ledger/posting", {
                "dateFrom": "2025-01-01",
                "dateTo": "2026-01-01",
                "accountNumberFrom": "1700",
                "accountNumberTo": "1700",
                "fields": "amountGross,account,date",
            })],
        )

    async def test_tripletex_api_call_flattens_invalid_ledger_posting_postings_field(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/posting",
                    (
                        ("accountNumberFrom", "1700"),
                        ("accountNumberTo", "1700"),
                        ("dateFrom", "2025-01-01"),
                        ("dateTo", "2026-01-01"),
                        ("fields", "amountGross,account,date,voucher(number)"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"amountGross": 27650, "voucher": {"number": 83}}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {
                "method": "GET",
                "path": "/ledger/posting?dateFrom=2025-01-01&dateTo=2026-01-01&accountNumberFrom=1700&accountNumberTo=1700&fields=amountGross,account,date,postings(voucher(number))",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["voucher"]["number"], 83)
        self.assertEqual(
            client.calls,
            [("GET", "/ledger/posting", {
                "dateFrom": "2025-01-01",
                "dateTo": "2026-01-01",
                "accountNumberFrom": "1700",
                "accountNumberTo": "1700",
                "fields": "amountGross,account,date,voucher(number)",
            })],
        )

    async def test_tripletex_api_call_normalizes_ledger_voucher_posting_voucher_number_field(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/voucher",
                    (
                        ("dateFrom", "2026-01-01"),
                        ("dateTo", "2026-03-01"),
                        ("fields", "id,date,description,postings(amountGross,account(number,name),voucher(number))"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 555, "description": "Voucher"}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {
                "method": "GET",
                "path": "/ledger/voucher?dateFrom=2026-01-01&dateTo=2026-03-01&fields=id,date,description,postings(amountGross,account(number,name),voucherNumber)",
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["id"], 555)
        self.assertEqual(
            client.calls,
            [("GET", "/ledger/voucher", {
                "dateFrom": "2026-01-01",
                "dateTo": "2026-03-01",
                "fields": "id,date,description,postings(amountGross,account(number,name),voucher(number))",
            })],
        )

    async def test_tripletex_api_call_blocks_ledger_result(self):
        client = FakeTripletexClient()

        with self.assertRaises(ValueError):
            await _execute(
                client,
                "tripletex_api_call",
                {"method": "GET", "path": "/ledger/result?dateFrom=2025-01-01&dateTo=2026-01-01"},
                endpoint_search=None,
                ctx=EntityContext(),
            )
        self.assertEqual(client.calls, [])

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

    async def test_tripletex_api_call_normalizes_supplier_invoice_payment_type_lookup(self):
        client = FakeTripletexClient(
            get_responses={
                ("/ledger/paymentTypeOut", ()): {"fullResultSize": 1, "values": [{"id": 7, "description": "Bank ut"}]},
            }
        )
        ctx = EntityContext()

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/supplierInvoice/paymentType"},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["values"][0]["id"], 7)
        self.assertEqual(ctx.last_payment_type_id, 7)
        self.assertEqual(client.calls, [("GET", "/ledger/paymentTypeOut", {})])

    async def test_tripletex_api_call_normalizes_invoice_child_fields_and_amount_remaining(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/invoice",
                    (
                        ("fields", "id,invoiceNumber,invoiceDate,amountOutstanding,customer(name)"),
                        ("invoiceDateFrom", "2026-01-01"),
                        ("invoiceDateTo", "2026-02-01"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 2147596454, "invoiceNumber": "2"}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/invoice?invoiceDateFrom=2026-01-01&invoiceDateTo=2026-02-01&fields=id,invoiceNumber,invoiceDate,amountRemaining,customer.name"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["id"], 2147596454)
        self.assertEqual(
            client.calls,
            [
                (
                    "GET",
                    "/invoice",
                    {
                        "invoiceDateFrom": "2026-01-01",
                        "invoiceDateTo": "2026-02-01",
                        "fields": "id,invoiceNumber,invoiceDate,amountOutstanding,customer(name)",
                    },
                )
            ],
        )

    async def test_tripletex_api_call_normalizes_supplier_invoice_fields_and_add_payment_params(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/supplierInvoice",
                    (
                        ("fields", "id,invoiceNumber,invoiceDate,amount,supplier(name)"),
                        ("invoiceDateFrom", "2026-01-01"),
                        ("invoiceDateTo", "2026-02-01"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 77, "invoiceNumber": "SI-77"}],
                }
            }
        )

        result = await _execute(
            client,
            "tripletex_api_call",
            {"method": "GET", "path": "/supplierInvoice?invoiceDateFrom=2026-01-01&invoiceDateTo=2026-02-01&fields=id,invoiceNumber,invoiceDate,amountRemaining,supplier.name"},
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(result["values"][0]["id"], 77)
        self.assertEqual(
            client.calls,
            [
                (
                    "GET",
                    "/supplierInvoice",
                    {
                        "invoiceDateFrom": "2026-01-01",
                        "invoiceDateTo": "2026-02-01",
                        "fields": "id,invoiceNumber,invoiceDate,amount,supplier(name)",
                    },
                )
            ],
        )

        client = FakeTripletexClient()

        ctx = EntityContext(prompt_text="Maneja los pagos parciales correctamente.")

        await _execute(
            client,
            "tripletex_api_call",
            {"method": "PUT", "path": "/supplierInvoice/77/:addPayment?paymentDate=2026-01-18&paymentTypeId=13&paidAmount=3650.00"},
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(
            client.calls,
            [
                (
                    "PUT",
                    "/supplierInvoice/77/:addPayment",
                    None,
                    {
                        "paymentDate": "2026-01-18",
                        "paymentType": "13",
                        "amount": "3650.00",
                        "partialPayment": True,
                    },
                )
            ],
        )
        self.assertEqual(ctx.supplier_invoice_payment_action_count, 1)

    async def test_tripletex_api_call_tracks_customer_invoice_payment_action(self):
        client = FakeTripletexClient(
            put_responses={
                "/invoice/2147596454/:payment": {"value": {"id": 2147596454}}
            }
        )
        ctx = EntityContext()

        result = await _execute(
            client,
            "tripletex_api_call",
            {
                "method": "PUT",
                "path": "/invoice/2147596454/:payment?paymentDate=2026-03-21&paymentTypeId=3&paidAmount=44100",
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 2147596454)
        self.assertEqual(ctx.last_invoice_id, 2147596454)
        self.assertEqual(ctx.invoice_payment_action_count, 1)
        self.assertEqual(ctx.customer_invoice_payment_action_count, 1)
        self.assertEqual(
            client.calls,
            [
                (
                    "PUT",
                    "/invoice/2147596454/:payment",
                    None,
                    {
                        "paymentDate": "2026-03-21",
                        "paymentTypeId": "3",
                        "paidAmount": "44100",
                    },
                )
            ],
        )

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

    async def test_create_salary_transaction_retries_after_auto_creating_missing_employment(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment",
                    (("count", 20), ("employeeId", 18610084), ("fields", "id,startDate,endDate")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/18610084",
                    (("fields", "id,firstName,lastName,email,dateOfBirth,department"),),
                ): {"value": {"id": 18610084, "dateOfBirth": "1990-05-01"}},
                (
                    "/division",
                    (("count", 1), ("fields", "id,name,organizationNumber")),
                ): {"fullResultSize": 1, "values": [{"id": 7001, "name": "Default business", "organizationNumber": "999888777"}]},
            },
            post_errors={
                "/salary/transaction": [
                    Exception("422 unknown: employee is not registered with an employment in the period"),
                    None,
                ],
            },
            post_responses={
                "/employee/employment": {"value": {"id": 2814181, "employee": {"id": 18610084}, "startDate": "2026-03-21"}},
            },
        )

        await _execute(
            client,
            "create_salary_transaction",
            {
                "date": "2026-03-21",
                "year": 2026,
                "month": 3,
                "payslips": [
                    {
                        "employee": {"id": 18610084},
                        "specifications": [
                            {"salaryType": {"id": 53154812}, "rate": 43050, "count": 1},
                        ],
                    }
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(
            client.calls,
            [
                ("POST", "/salary/transaction", {
                    "date": "2026-03-21",
                    "year": 2026,
                    "month": 3,
                    "payslips": [
                        {
                            "employee": {"id": 18610084},
                            "specifications": [
                                {"salaryType": {"id": 53154812}, "rate": 43050, "count": 1},
                            ],
                        }
                    ],
                }),
                ("GET", "/employee/employment", {"employeeId": 18610084, "fields": "id,startDate,endDate", "count": 20}),
                ("GET", "/employee/18610084", {"fields": "id,firstName,lastName,email,dateOfBirth,department"}),
                ("GET", "/division", {"fields": "id,name,organizationNumber", "count": 1}),
                ("POST", "/employee/employment", {"employee": {"id": 18610084, "dateOfBirth": "1990-05-01"}, "startDate": "2026-03-21", "division": {"id": 7001}}),
                ("POST", "/salary/transaction", {
                    "date": "2026-03-21",
                    "year": 2026,
                    "month": 3,
                    "payslips": [
                        {
                            "employee": {"id": 18610084},
                            "specifications": [
                                {"salaryType": {"id": 53154812}, "rate": 43050, "count": 1},
                            ],
                        }
                    ],
                }),
            ],
        )

    async def test_create_salary_transaction_retries_after_setting_placeholder_dob(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment",
                    (("count", 20), ("employeeId", 18610434), ("fields", "id,startDate,endDate")),
                ): {"fullResultSize": 0, "values": []},
                (
                    "/employee/18610434",
                    (("fields", "id,firstName,lastName,email,dateOfBirth,department"),),
                ): {"value": {"id": 18610434, "email": "ana.ferreira@example.org"}},
                (
                    "/division",
                    (("count", 1), ("fields", "id,name,organizationNumber")),
                ): {"fullResultSize": 1, "values": [{"id": 7001, "name": "Default business", "organizationNumber": "999888777"}]},
            },
            post_errors={
                "/salary/transaction": [
                    Exception("422 unknown: employee is not registered with an employment in the period"),
                    None,
                ],
            },
            post_responses={
                "/employee/employment": {"value": {"id": 2815301, "employee": {"id": 18610434}, "startDate": "2026-03-21"}},
            },
        )

        await _execute(
            client,
            "create_salary_transaction",
            {
                "date": "2026-03-21",
                "year": 2026,
                "month": 3,
                "payslips": [
                    {
                        "employee": {"id": 18610434},
                        "specifications": [
                            {"salaryType": {"id": 53258510}, "rate": 41750, "count": 1},
                            {"salaryType": {"id": 53258687}, "rate": 6750, "count": 1},
                        ],
                    }
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(
            client.calls,
            [
                ("POST", "/salary/transaction", {
                    "date": "2026-03-21",
                    "year": 2026,
                    "month": 3,
                    "payslips": [
                        {
                            "employee": {"id": 18610434},
                            "specifications": [
                                {"salaryType": {"id": 53258510}, "rate": 41750, "count": 1},
                                {"salaryType": {"id": 53258687}, "rate": 6750, "count": 1},
                            ],
                        }
                    ],
                }),
                ("GET", "/employee/employment", {"employeeId": 18610434, "fields": "id,startDate,endDate", "count": 20}),
                ("GET", "/employee/18610434", {"fields": "id,firstName,lastName,email,dateOfBirth,department"}),
                ("PUT", "/employee/18610434", {"id": 18610434, "dateOfBirth": "1990-01-01"}, None),
                ("GET", "/division", {"fields": "id,name,organizationNumber", "count": 1}),
                ("POST", "/employee/employment", {"employee": {"id": 18610434, "dateOfBirth": "1990-01-01"}, "startDate": "2026-03-21", "division": {"id": 7001}}),
                ("POST", "/salary/transaction", {
                    "date": "2026-03-21",
                    "year": 2026,
                    "month": 3,
                    "payslips": [
                        {
                            "employee": {"id": 18610434},
                            "specifications": [
                                {"salaryType": {"id": 53258510}, "rate": 41750, "count": 1},
                                {"salaryType": {"id": 53258687}, "rate": 6750, "count": 1},
                            ],
                        }
                    ],
                }),
            ],
        )

    async def test_create_salary_transaction_retries_after_linking_employment_to_division(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/employee/employment",
                    (("count", 20), ("employeeId", 18613724), ("fields", "id,startDate,endDate")),
                ): {"fullResultSize": 1, "values": [{"id": 2816401, "startDate": "2026-03-21"}]},
                (
                    "/division",
                    (("count", 1), ("fields", "id,name,organizationNumber")),
                ): {"fullResultSize": 1, "values": [{"id": 7001, "name": "Default business", "organizationNumber": "999888777"}]},
                (
                    "/employee/employment/2816401",
                    (("fields", "id,division"),),
                ): {"value": {"id": 2816401}},
            },
            post_errors={
                "/salary/transaction": [
                    Exception("422 unknown: Arbeidsforholdet er ikke knyttet mot en virksomhet."),
                    None,
                ],
            },
        )

        await _execute(
            client,
            "create_salary_transaction",
            {
                "date": "2026-03-21",
                "year": 2026,
                "month": 3,
                "payslips": [
                    {
                        "employee": {"id": 18613724},
                        "specifications": [
                            {"salaryType": {"id": 54182004}, "rate": 58750, "count": 1},
                            {"salaryType": {"id": 54182165}, "rate": 10750, "count": 1},
                        ],
                    }
                ],
            },
            endpoint_search=None,
            ctx=EntityContext(),
        )

        self.assertEqual(
            client.calls,
            [
                ("POST", "/salary/transaction", {
                    "date": "2026-03-21",
                    "year": 2026,
                    "month": 3,
                    "payslips": [
                        {
                            "employee": {"id": 18613724},
                            "specifications": [
                                {"salaryType": {"id": 54182004}, "rate": 58750, "count": 1},
                                {"salaryType": {"id": 54182165}, "rate": 10750, "count": 1},
                            ],
                        }
                    ],
                }),
                ("GET", "/employee/employment", {"employeeId": 18613724, "fields": "id,startDate,endDate", "count": 20}),
                ("GET", "/division", {"fields": "id,name,organizationNumber", "count": 1}),
                ("GET", "/employee/employment/2816401", {"fields": "id,division"}),
                ("PUT", "/employee/employment/2816401", {"id": 2816401, "division": {"id": 7001}}, None),
                ("POST", "/salary/transaction", {
                    "date": "2026-03-21",
                    "year": 2026,
                    "month": 3,
                    "payslips": [
                        {
                            "employee": {"id": 18613724},
                            "specifications": [
                                {"salaryType": {"id": 54182004}, "rate": 58750, "count": 1},
                                {"salaryType": {"id": 54182165}, "rate": 10750, "count": 1},
                            ],
                        }
                    ],
                }),
            ],
        )

    async def test_find_top_expense_account_increases_blocks_identical_repeat(self):
        client = FakeTripletexClient()
        ctx = EntityContext()

        first = await _execute(
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
            ctx=ctx,
        )
        second = await _execute(
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
            ctx=ctx,
        )

        self.assertIn("topAccounts", first)
        self.assertIn("already ran for this exact comparison", second["error"])
        self.assertEqual(len(client.calls), 2)

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

    async def test_create_voucher_retries_with_customer_on_credit_receivable_posting(self):
        client = FakeTripletexClient(
            post_errors={
                "/ledger/voucher": [
                    Exception("422 Validation failed: Kunde mangler."),
                    None,
                ]
            }
        )
        ctx = EntityContext(last_customer_id=108342671)
        ctx.account_cache = {
            8160: {"id": 8160, "number": 8160, "name": "Valutatap"},
            1500: {"id": 1500, "number": 1500, "name": "Kundefordringer"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Valutatap",
                "postings": [
                    {"account": {"id": 8160}, "amountGross": 8977.17},
                    {"account": {"id": 1500}, "amountGross": -8977.17},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertNotIn("customer", client.calls[1][2]["postings"][0])
        self.assertEqual(client.calls[1][2]["postings"][1]["customer"], {"id": 108342671})

    async def test_create_voucher_allows_supplier_invoice_auto_vat_balance(self):
        client = FakeTripletexClient()
        ctx = EntityContext()
        ctx.account_cache = {
            10: {
                "id": 10,
                "number": 6700,
                "name": "Vedlikehold",
                "vatType": {"id": 50},
                "legalVatTypes": [{"id": 50}],
                "vatLocked": False,
            },
            20: {
                "id": 20,
                "number": 2400,
                "name": "Leverandorgjeld",
            },
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-05-22",
                "description": "Supplier invoice",
                "postings": [
                    {"account": {"id": 10}, "amountGross": 65450, "vatType": {"id": 50}},
                    {"account": {"id": 20}, "amountGross": -81812, "supplier": {"id": 77}},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][0:2], ("POST", "/ledger/voucher"))
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGross"], 81812)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGrossCurrency"], 81812)
        self.assertEqual(client.calls[-1][2]["postings"][0]["vatType"], {"id": 50})

    async def test_create_voucher_expands_simple_supplier_invoice_gross_amount_into_explicit_vat_split(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (
                        ("fields", "id,number,name,vatType,vatLocked,requiresDepartment,legalVatTypes,isApplicableForSupplierInvoice,isBankAccount"),
                        ("number", "2710"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 30, "number": 2710, "name": "Inngaaende mva"}],
                }
            }
        )
        ctx = EntityContext(last_department_id=730036)
        ctx.account_cache = {
            10: {
                "id": 10,
                "number": 6590,
                "name": "Burodienstleistungen",
                "vatType": {"id": 1},
                "legalVatTypes": [{"id": 1}],
                "vatLocked": False,
            },
            20: {
                "id": 20,
                "number": 2400,
                "name": "Leverandorgjeld",
            },
        }
        ctx.vat_type_cache = {(25.0, "incoming"): 1}

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Supplier invoice INV-2026-2118",
                "postings": [
                    {
                        "account": {"id": 10},
                        "amountGross": 70400,
                        "amountGrossCurrency": 70400,
                        "vatType": {"id": 1},
                        "description": "Burodienstleistungen",
                    },
                    {
                        "account": {"id": 20},
                        "amountGross": -70400,
                        "amountGrossCurrency": -70400,
                        "supplier": {"id": 77},
                        "description": "Supplier invoice",
                    },
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[0][0:2], ("GET", "/ledger/account"))
        self.assertEqual(client.calls[-1][0:2], ("POST", "/ledger/voucher"))
        voucher_postings = client.calls[-1][2]["postings"]
        self.assertEqual(len(voucher_postings), 3)
        self.assertEqual(voucher_postings[0]["amountGross"], 56320.0)
        self.assertNotIn("vatType", voucher_postings[0])
        self.assertEqual(voucher_postings[1]["account"], {"id": 30})
        self.assertEqual(voucher_postings[1]["amountGross"], 14080.0)
        self.assertEqual(voucher_postings[1]["department"], {"id": 730036})
        self.assertEqual(voucher_postings[2]["amountGross"], -70400)

    async def test_create_voucher_keeps_supplier_reference_only_on_payables_line(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (
                        ("fields", "id,number,name,vatType,vatLocked,requiresDepartment,legalVatTypes,isApplicableForSupplierInvoice,isBankAccount"),
                        ("number", "2710"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 30, "number": 2710, "name": "Inngaaende mva"}],
                }
            }
        )
        ctx = EntityContext(last_department_id=933349)
        ctx.account_cache = {
            10: {
                "id": 10,
                "number": 6790,
                "name": "Annen fremmed tjeneste",
                "vatType": {"id": 1},
                "legalVatTypes": [{"id": 1}],
                "vatLocked": False,
            },
            20: {
                "id": 20,
                "number": 2400,
                "name": "Leverandorgjeld",
            },
        }
        ctx.vat_type_cache = {(25.0, "incoming"): 1}

        await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Leverandorkostnad prosjekt",
                "postings": [
                    {
                        "account": {"id": 10},
                        "amountGross": 23500,
                        "amountGrossCurrency": 23500,
                        "vatType": {"id": 1},
                        "supplier": {"id": 108353437},
                        "project": {"id": 402008098},
                    },
                    {
                        "account": {"id": 20},
                        "amountGross": -23500,
                        "amountGrossCurrency": -23500,
                        "supplier": {"id": 108353437},
                    },
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        voucher_postings = client.calls[-1][2]["postings"]
        self.assertNotIn("supplier", voucher_postings[0])
        self.assertNotIn("supplier", voucher_postings[1])
        self.assertEqual(voucher_postings[2]["supplier"], {"id": 108353437})

    async def test_create_voucher_normalizes_cloud_supplier_invoice_account_from_6340_to_6420(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (
                        ("fields", "id,number,name,vatType,vatLocked,requiresDepartment,legalVatTypes,isApplicableForSupplierInvoice,isBankAccount"),
                        ("number", "6420"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 40, "number": 6420, "name": "Programvare"}],
                },
                (
                    "/ledger/account",
                    (
                        ("fields", "id,number,name,vatType,vatLocked,requiresDepartment,legalVatTypes,isApplicableForSupplierInvoice,isBankAccount"),
                        ("number", "2710"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 30, "number": 2710, "name": "Inngaaende mva"}],
                },
            }
        )
        ctx = EntityContext(last_department_id=708468)
        ctx.account_cache = {
            10: {
                "id": 10,
                "number": 6340,
                "name": "Lys og varme",
                "vatType": {"id": 1},
                "legalVatTypes": [{"id": 1}],
                "vatLocked": False,
            },
            20: {
                "id": 20,
                "number": 2400,
                "name": "Leverandorgjeld",
            },
        }
        ctx.vat_type_cache = {(25.0, "incoming"): 1}

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-05",
                "description": "Supplier invoice INV-2026-4660 Greenfield Ltd - Cloud storage",
                "postings": [
                    {
                        "account": {"id": 10},
                        "amountGross": 52437,
                        "amountGrossCurrency": 52437,
                        "vatType": {"id": 1},
                        "description": "Cloud storage",
                    },
                    {
                        "account": {"id": 20},
                        "amountGross": -52437,
                        "amountGrossCurrency": -52437,
                        "supplier": {"id": 108350020},
                        "description": "Supplier invoice INV-2026-4660",
                    },
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[0][0:2], ("GET", "/ledger/account"))
        self.assertEqual(client.calls[1][0:2], ("GET", "/ledger/account"))
        voucher_postings = client.calls[-1][2]["postings"]
        self.assertEqual(voucher_postings[0]["account"], {"id": 40})
        self.assertEqual(voucher_postings[0]["amountGross"], 41949.6)
        self.assertEqual(voucher_postings[1]["account"], {"id": 30})
        self.assertEqual(voucher_postings[1]["amountGross"], 10487.4)
        self.assertEqual(voucher_postings[2]["amountGross"], -52437)

    async def test_create_voucher_normalizes_year_end_depreciation_to_requested_accounts(self):
        client = FakeTripletexClient()
        ctx = EntityContext()
        ctx.account_cache = {
            1: {"id": 1, "number": 6010, "name": "Avskrivning"},
            2: {"id": 2, "number": 1209, "name": "Akkumulert avskrivning"},
            3: {"id": 3, "number": 1259, "name": "Akkumulert avskrivning programvare"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Årsavskrivning 2025 - Programvare",
                "postings": [
                    {"account": {"id": 1}, "amountGross": 123983.33, "amountGrossCurrency": 123983.33},
                    {"account": {"id": 3}, "amountGross": -123983.33, "amountGrossCurrency": -123983.33},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][2]["postings"][0]["account"], {"id": 1})
        self.assertEqual(client.calls[-1][2]["postings"][1]["account"], {"id": 2})

    async def test_create_voucher_normalizes_german_year_end_depreciation_to_requested_accounts(self):
        client = FakeTripletexClient()
        ctx = EntityContext()
        ctx.account_cache = {
            1: {"id": 1, "number": 6010, "name": "Avskrivning"},
            2: {"id": 2, "number": 1209, "name": "Akkumulert avskrivning"},
            3: {"id": 3, "number": 1249, "name": "Akkumulert avskrivning inventar"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Jaehrliche Abschreibung 2025 - Inventar",
                "postings": [
                    {"account": {"id": 1}, "amountGross": 47550, "amountGrossCurrency": 47550},
                    {"account": {"id": 3}, "amountGross": -47550, "amountGrossCurrency": -47550},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][2]["postings"][0]["account"], {"id": 1})
        self.assertEqual(client.calls[-1][2]["postings"][1]["account"], {"id": 2})

    async def test_create_voucher_normalizes_year_end_prepaid_reversal_to_prompt_total(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Realize o encerramento anual simplificado de 2025. "
                "Reverta despesas antecipadas (total 63300 NOK na conta 1700)."
            ),
        )
        ctx.account_cache = {
            1: {"id": 1, "number": 6300, "name": "Leie lokale"},
            2: {"id": 2, "number": 1700, "name": "Forskuddsbetalte kostnader"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Tilbakeføring forskuddsbetalt leiekostnad 2025",
                "postings": [
                    {"account": {"id": 1}, "amountGross": 5275, "amountGrossCurrency": 5275},
                    {"account": {"id": 2}, "amountGross": -5275, "amountGrossCurrency": -5275},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGross"], 63300)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGrossCurrency"], 63300)
        self.assertEqual(client.calls[-1][2]["postings"][1]["amountGross"], -63300)
        self.assertEqual(client.calls[-1][2]["postings"][1]["amountGrossCurrency"], -63300)

    async def test_create_voucher_normalizes_french_year_end_prepaid_reversal_to_prompt_total(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Effectuez la clôture annuelle simplifiée pour 2025. "
                "Extournez les charges constatées d'avance (total 23750 NOK au compte 1700)."
            ),
        )
        ctx.account_cache = {
            1: {"id": 1, "number": 6300, "name": "Leie lokale"},
            2: {"id": 2, "number": 1700, "name": "Forskuddsbetalte kostnader"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Extourne charges constatées d'avance 2025",
                "postings": [
                    {"account": {"id": 1}, "amountGross": 1979.13, "amountGrossCurrency": 1979.13},
                    {"account": {"id": 2}, "amountGross": -1979.13, "amountGrossCurrency": -1979.13},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGross"], 23750)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGrossCurrency"], 23750)
        self.assertEqual(client.calls[-1][2]["postings"][1]["amountGross"], -23750)
        self.assertEqual(client.calls[-1][2]["postings"][1]["amountGrossCurrency"], -23750)

    async def test_create_voucher_normalizes_german_year_end_prepaid_reversal_to_prompt_total(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Fuehren Sie den vereinfachten Jahresabschluss fuer 2025 durch. "
                "Loesen Sie vorausbezahlte Aufwendungen auf (insgesamt 73050 NOK auf Konto 1700). "
                "3) Berechnen und buchen Sie die Steuerrueckstellung."
            ),
        )
        ctx.account_cache = {
            1: {"id": 1, "number": 6300, "name": "Leie lokale"},
            2: {"id": 2, "number": 1700, "name": "Forskuddsbetalte kostnader"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Aufloesung vorausbezahlte Aufwendungen 2025",
                "postings": [
                    {"account": {"id": 1}, "amountGross": 6087.5, "amountGrossCurrency": 6087.5},
                    {"account": {"id": 2}, "amountGross": -6087.5, "amountGrossCurrency": -6087.5},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGross"], 73050)
        self.assertEqual(client.calls[-1][2]["postings"][0]["amountGrossCurrency"], 73050)
        self.assertEqual(client.calls[-1][2]["postings"][1]["amountGross"], -73050)
        self.assertEqual(client.calls[-1][2]["postings"][1]["amountGrossCurrency"], -73050)

    async def test_create_voucher_normalizes_year_end_tax_provision_to_prompt_accounts(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Fuehren Sie den vereinfachten Jahresabschluss fuer 2025 durch. "
                "Berechnen und buchen Sie die Steuerrueckstellung auf Konto 8700/2920."
            ),
        )
        ctx.account_cache = {
            1: {"id": 1, "number": 8700, "name": "Skattekostnad"},
            2: {"id": 2, "number": 2920, "name": "Avsetning skatt"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Steuerrueckstellung / Skattekostnad 2025",
                "postings": [
                    {"account": {"id": 1}, "amountGross": 994136.13, "amountGrossCurrency": 994136.13},
                    {"account": {"id": 1}, "amountGross": -994136.13, "amountGrossCurrency": -994136.13},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(client.calls[-1][2]["postings"][0]["account"], {"id": 1})
        self.assertEqual(client.calls[-1][2]["postings"][1]["account"], {"id": 2})

    async def test_create_voucher_normalizes_year_end_prepaid_rent_counterpart_to_6300(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/ledger/account",
                    (("fields", "id,number,name"), ("number", "6300")),
                ): {
                    "fullResultSize": 1,
                    "values": [{"id": 3, "number": 6300, "name": "Leie lokaler"}],
                }
            }
        )
        ctx = EntityContext(
            prompt_text=(
                "Fuehren Sie den vereinfachten Jahresabschluss fuer 2025 durch. "
                "Loesen Sie vorausbezahlte Aufwendungen auf (insgesamt 27650 NOK auf Konto 1700)."
            ),
        )
        ctx.account_cache = {
            1: {"id": 1, "number": 1700, "name": "Forskuddsbetalte kostnader"},
            2: {"id": 2, "number": 1209, "name": "Akkumulert avskrivning"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2025-12-31",
                "description": "Oppløsning forskuddsbetalt kostnad 2025",
                "postings": [
                    {
                        "account": {"id": 1},
                        "amountGross": 27650,
                        "amountGrossCurrency": 27650,
                        "description": "Oppløsning forskuddsbetalt leie",
                    },
                    {
                        "account": {"id": 2},
                        "amountGross": -27650,
                        "amountGrossCurrency": -27650,
                        "description": "Motkonto forskuddsbetalt leie",
                    },
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual(result["value"]["id"], 999)
        self.assertEqual(
            client.calls,
            [
                ("GET", "/ledger/account", {"number": "6300", "fields": "id,number,name"}),
                (
                    "POST",
                    "/ledger/voucher",
                    {
                        "date": "2025-12-31",
                        "description": "Oppløsning forskuddsbetalt kostnad 2025",
                        "postings": [
                            {
                                "account": {"id": 1},
                                "amountGross": 27650,
                                "amountGrossCurrency": 27650,
                                "description": "Oppløsning forskuddsbetalt leie",
                                "row": 1,
                            },
                            {
                                "account": {"id": 3},
                                "amountGross": -27650,
                                "amountGrossCurrency": -27650,
                                "description": "Motkonto forskuddsbetalt leie",
                                "row": 2,
                            },
                        ],
                    },
                ),
            ],
        )

    async def test_create_voucher_splits_month_end_closing_and_normalizes_requested_accounts(self):
        client = FakeTripletexClient()
        ctx = EntityContext(last_department_id=931987)
        ctx.account_cache = {
            10: {"id": 10, "number": 8300, "name": "Wrongly guessed expense"},
            11: {"id": 11, "number": 1720, "name": "Forskuddsbetalte kostnader"},
            12: {"id": 12, "number": 6010, "name": "Avskrivning"},
            13: {"id": 13, "number": 1239, "name": "Akkumulert avskrivning"},
            14: {"id": 14, "number": 6030, "name": "Avskrivning transportmidler"},
            19: {"id": 19, "number": 1209, "name": "Akkumulert avskrivning"},
            15: {"id": 15, "number": 5020, "name": "Wrong salary expense"},
            16: {"id": 16, "number": 2990, "name": "Wrong accrued salary"},
            17: {"id": 17, "number": 5000, "name": "Lonn til ansatte"},
            18: {"id": 18, "number": 2900, "name": "Skyldig lonn"},
        }

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-31",
                "description": "Month-end closing March 2026 - accrual reversal, depreciation, and salary accrual",
                "year": 2026,
                "postings": [
                    {"account": {"id": 10}, "amountGross": 8300, "amountGrossCurrency": 8300, "description": "Accrual reversal to expense for March from 1720"},
                    {"account": {"id": 11}, "amountGross": -8300, "amountGrossCurrency": -8300, "description": "Reverse monthly accrual from 1720"},
                    {"account": {"id": 12}, "amountGross": 1699.58, "amountGrossCurrency": 1699.58, "description": "Monthly depreciation March 2026"},
                    {"account": {"id": 13}, "amountGross": -1699.58, "amountGrossCurrency": -1699.58, "description": "Accumulated depreciation March 2026"},
                    {"account": {"id": 15}, "amountGross": 45000, "amountGrossCurrency": 45000, "description": "Salary accrual March 2026"},
                    {"account": {"id": 16}, "amountGross": -45000, "amountGrossCurrency": -45000, "description": "Accrued salary March 2026"},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertEqual([call[0:2] for call in client.calls], [("POST", "/ledger/voucher")] * 3)
        self.assertEqual(client.calls[0][2]["description"], "Month-end closing March 2026 - accrual reversal")
        self.assertEqual(client.calls[1][2]["description"], "Month-end closing March 2026 - depreciation")
        self.assertEqual(client.calls[2][2]["description"], "Month-end closing March 2026 - salary accrual")
        self.assertEqual(client.calls[1][2]["postings"][0]["account"], {"id": 14})
        self.assertEqual(client.calls[1][2]["postings"][1]["account"], {"id": 19})
        self.assertEqual(client.calls[2][2]["postings"][0]["account"], {"id": 17})
        self.assertEqual(client.calls[2][2]["postings"][1]["account"], {"id": 18})
        self.assertEqual(len(result["values"]), 3)

    async def test_create_voucher_normalizes_german_month_end_accrual_reversal_accounts(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Führen Sie den Monatsabschluss für März 2026 durch. "
                "Buchen Sie die Rechnungsabgrenzung (4200 NOK pro Monat von Konto 1700 auf Aufwand)."
            )
        )
        ctx.account_cache = {
            1: {"id": 1, "number": 4200, "name": "Amount mistaken for account"},
            2: {"id": 2, "number": 1700, "name": "Forskuddsbetalte kostnader"},
            3: {"id": 3, "number": 6000, "name": "Annen driftskostnad"},
            4: {"id": 4, "number": 1209, "name": "Akkumulert avskrivning"},
        }

        await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-31",
                "description": "Monatsabschluss März 2026 - Rechnungsabgrenzung",
                "year": 2026,
                "postings": [
                    {"account": {"id": 1}, "amountGross": 4200, "amountGrossCurrency": 4200},
                    {"account": {"id": 4}, "amountGross": -4200, "amountGrossCurrency": -4200},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/ledger/voucher"))
        self.assertEqual(body["postings"][0]["account"], {"id": 3})
        self.assertEqual(body["postings"][1]["account"], {"id": 2})

    async def test_create_voucher_keeps_supported_locked_vattype(self):
        client = FakeTripletexClient()
        ctx = EntityContext()
        ctx.account_cache = {
            10: {
                "id": 10,
                "number": 6300,
                "name": "Vedlikeholdsavtale",
                "vatType": {"id": 50},
                "legalVatTypes": [{"id": 50}],
                "vatLocked": True,
            },
            20: {
                "id": 20,
                "number": 2400,
                "name": "Leverandorgjeld",
            },
        }

        await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-05-22",
                "description": "Supplier invoice",
                "postings": [
                    {"account": {"id": 10}, "amountGross": 65450, "vatType": {"id": 1}},
                    {"account": {"id": 20}, "amountGross": -81812, "supplier": {"id": 77}},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        method, path, body = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/ledger/voucher"))
        self.assertEqual(body["postings"][0]["vatType"], {"id": 50})

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

    async def test_create_travel_expense_auto_populates_domestic_rate_based_details(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_expense",
            {
                "employee": {"id": 42},
                "title": "Visite client Oslo",
                "departureDate": "2026-03-17",
                "returnDate": "2026-03-21",
            },
            endpoint_search=None,
            ctx=EntityContext(
                prompt_text="Enregistrez une note de frais de déplacement avec indemnités journalières pour Visite client Oslo.",
            ),
        )

        self.assertEqual(client.calls, [("POST", "/travelExpense", {
            "employee": {"id": 42},
            "title": "Visite client Oslo",
            "travelDetails": {
                "departureDate": "2026-03-17",
                "returnDate": "2026-03-21",
                "destination": "Oslo",
                "purpose": "Visite client Oslo",
                "isDayTrip": False,
                "isCompensationFromRates": True,
                "isForeignTravel": False,
            },
        })])

    async def test_create_travel_expense_sets_is_compensation_from_rates_for_portuguese_allowance_prompt(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_expense",
            {
                "employee": {"id": 42},
                "title": "Visita cliente Trondheim",
                "departureDate": "2026-03-17",
                "returnDate": "2026-03-21",
            },
            endpoint_search=None,
            ctx=EntityContext(
                prompt_text="Registe uma despesa de viagem com ajudas de custo para Visita cliente Trondheim.",
            ),
        )

        self.assertTrue(client.calls[0][2]["travelDetails"]["isCompensationFromRates"])

    async def test_create_travel_expense_sets_is_compensation_from_rates_for_spanish_dietas_prompt(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_expense",
            {
                "employee": {"id": 42},
                "title": "Visita cliente Oslo",
                "departureDate": "2026-03-23",
                "returnDate": "2026-03-24",
            },
            endpoint_search=None,
            ctx=EntityContext(
                prompt_text="Registre una nota de gastos de viaje para Visita cliente Oslo. El viaje duró 2 días con dietas (tarifa diaria 800 NOK).",
            ),
        )

        self.assertTrue(client.calls[0][2]["travelDetails"]["isCompensationFromRates"])

    async def test_create_travel_expense_shifts_undated_weekend_window_to_next_working_days(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_expense",
            {
                "employee": {"id": 42},
                "title": "Conference Bergen",
                "departureDate": "2026-03-21",
                "returnDate": "2026-03-22",
            },
            endpoint_search=None,
            ctx=EntityContext(
                prompt_text="Register a travel expense for Conference Bergen. The trip lasted 2 days with per diem.",
            ),
        )

        self.assertEqual(client.calls, [("POST", "/travelExpense", {
            "employee": {"id": 42},
            "title": "Conference Bergen",
            "travelDetails": {
                "departureDate": "2026-03-23",
                "returnDate": "2026-03-24",
                "destination": "Bergen",
                "purpose": "Conference Bergen",
                "isDayTrip": False,
                "isCompensationFromRates": True,
                "isForeignTravel": False,
            },
        })])

    async def test_create_per_diem_infers_no_and_resolves_rate_category_from_dates(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/travelExpense/rateCategory",
                    (
                        ("count", 100),
                        ("dateFrom", "2026-03-19"),
                        ("dateTo", "2026-03-22"),
                        ("isValidAccommodation", True),
                        ("isValidDomestic", True),
                        ("type", "PER_DIEM"),
                    ),
                ): {
                    "fullResultSize": 2,
                    "values": [
                        {
                            "id": 2,
                            "name": "Old category",
                            "type": "PER_DIEM",
                            "isValidAccommodation": True,
                            "isValidDomestic": True,
                            "isRequiresOvernightAccommodation": True,
                            "fromDate": "2025-01-01",
                            "toDate": "2026-01-01",
                        },
                        {
                            "id": 740,
                            "name": "Current category",
                            "type": "PER_DIEM",
                            "isValidAccommodation": True,
                            "isValidDomestic": True,
                            "isRequiresOvernightAccommodation": True,
                            "fromDate": "2026-01-01",
                            "toDate": "2027-01-01",
                        },
                    ],
                },
            }
        )

        await _execute(
            client,
            "create_per_diem_compensation",
            {
                "location": "Tromsø",
                "overnightAccommodation": "HOTEL",
                "count": 3,
                "rate": 800,
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_id=555,
                last_rate_category_id=2,
                last_travel_expense_departure_date="2026-03-19",
                last_travel_expense_return_date="2026-03-21",
                prompt_text='Erfassen Sie eine Reisekostenabrechnung für "Kundenbesuch Tromsø".',
            ),
        )

        self.assertEqual(client.calls[0], ("GET", "/travelExpense/rateCategory", {
            "type": "PER_DIEM",
            "count": 100,
            "dateFrom": "2026-03-19",
            "dateTo": "2026-03-22",
            "isValidAccommodation": True,
            "isValidDomestic": True,
        }))
        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/perDiemCompensation", {
            "travelExpense": {"id": 555},
            "rateCategory": {"id": 740},
            "location": "Tromsø",
            "overnightAccommodation": "HOTEL",
            "count": 3,
            "rate": 800,
        }))

    async def test_create_per_diem_retries_with_resolved_rate_category_after_validation_error(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/travelExpense/rateCategory",
                    (
                        ("count", 100),
                        ("dateFrom", "2026-03-19"),
                        ("dateTo", "2026-03-22"),
                        ("isValidAccommodation", True),
                        ("isValidDomestic", True),
                        ("type", "PER_DIEM"),
                    ),
                ): {
                    "fullResultSize": 1,
                    "values": [
                        {
                            "id": 740,
                            "name": "Current category",
                            "type": "PER_DIEM",
                            "isValidAccommodation": True,
                            "isValidDomestic": True,
                            "isRequiresOvernightAccommodation": True,
                            "fromDate": "2026-01-01",
                            "toDate": "2027-01-01",
                        },
                    ],
                },
            },
            post_errors={
                "/travelExpense/perDiemCompensation": [
                    self._http_status_error(
                        "/travelExpense/perDiemCompensation",
                        "422 unknown: rateCategory.id Reiseregningens dato samsvarer ikke med valgt satskategori.",
                    ),
                    None,
                ],
            },
        )

        await _execute(
            client,
            "create_per_diem_compensation",
            {
                "travelExpense": {"id": 555},
                "countryCode": "NO",
                "rateCategory": {"id": 2},
                "location": "Tromsø",
                "overnightAccommodation": "HOTEL",
                "count": 3,
                "rate": 800,
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_departure_date="2026-03-19",
                last_travel_expense_return_date="2026-03-21",
                prompt_text='Erfassen Sie eine Reisekostenabrechnung für "Kundenbesuch Tromsø".',
            ),
        )

        self.assertEqual(client.calls[0], ("POST", "/travelExpense/perDiemCompensation", {
            "travelExpense": {"id": 555},
            "rateCategory": {"id": 2},
            "location": "Tromsø",
            "overnightAccommodation": "HOTEL",
            "count": 3,
            "rate": 800,
        }))
        self.assertEqual(client.calls[1], ("GET", "/travelExpense/rateCategory", {
            "type": "PER_DIEM",
            "count": 100,
            "dateFrom": "2026-03-19",
            "dateTo": "2026-03-22",
            "isValidAccommodation": True,
            "isValidDomestic": True,
        }))
        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/perDiemCompensation", {
            "travelExpense": {"id": 555},
            "rateCategory": {"id": 740},
            "location": "Tromsø",
            "overnightAccommodation": "HOTEL",
            "count": 3,
            "rate": 800,
        }))


    async def test_create_per_diem_omits_explicit_domestic_country_code_before_first_post(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_per_diem_compensation",
            {
                "travelExpense": {"id": 555},
                "countryCode": "NO",
                "rateCategory": {"id": 740},
                "location": "Oslo",
                "overnightAccommodation": "HOTEL",
                "count": 5,
                "rate": 800,
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_departure_date="2026-03-17",
                last_travel_expense_return_date="2026-03-21",
                prompt_text='Enregistrez une note de frais pour "Visite client Oslo".',
            ),
        )

        self.assertEqual(client.calls, [("POST", "/travelExpense/perDiemCompensation", {
            "travelExpense": {"id": 555},
            "rateCategory": {"id": 740},
            "location": "Oslo",
            "overnightAccommodation": "HOTEL",
            "count": 5,
            "rate": 800,
        })])

    async def test_create_travel_cost_resolves_flight_category_and_infers_departure_date(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_cost",
            {
                "comments": "Bilhete de avião",
                "amountCurrencyIncVat": 7600,
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_id=555,
                last_payment_type_id=9,
                last_travel_expense_departure_date="2026-03-17",
                last_travel_expense_return_date="2026-03-21",
                last_cost_categories=[
                    {"id": 11, "displayName": "Flyreise"},
                    {"id": 12, "displayName": "Taxi"},
                ],
            ),
        )

        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/cost", {
            "travelExpense": {"id": 555},
            "costCategory": {"id": 11},
            "paymentType": {"id": 9},
            "comments": "Bilhete de avião",
            "amountCurrencyIncVat": 7600,
            "date": "2026-03-17",
        }))

    async def test_create_travel_cost_resolves_spanish_flight_category(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_cost",
            {
                "comments": "Billete de avión",
                "amountCurrencyIncVat": 3700,
                "date": "2026-03-23",
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_id=555,
                last_payment_type_id=9,
                last_cost_categories=[
                    {"id": 11, "displayName": "Flyreise"},
                    {"id": 12, "displayName": "Taxi"},
                ],
            ),
        )

        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/cost", {
            "travelExpense": {"id": 555},
            "costCategory": {"id": 11},
            "paymentType": {"id": 9},
            "comments": "Billete de avión",
            "amountCurrencyIncVat": 3700,
            "date": "2026-03-23",
        }))

    async def test_create_travel_cost_aligns_undated_flight_date_to_normalized_departure(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_cost",
            {
                "comments": "Flight ticket",
                "amountCurrencyIncVat": 6400,
                "date": "2026-03-21",
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_id=555,
                last_payment_type_id=9,
                last_travel_expense_departure_date="2026-03-23",
                last_travel_expense_return_date="2026-03-24",
                last_cost_categories=[
                    {"id": 11, "displayName": "Flight"},
                    {"id": 12, "displayName": "Taxi"},
                ],
                prompt_text="Register a travel expense for Conference Bergen. The trip lasted 2 days with per diem.",
            ),
        )

        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/cost", {
            "travelExpense": {"id": 555},
            "costCategory": {"id": 11},
            "paymentType": {"id": 9},
            "comments": "Flight ticket",
            "amountCurrencyIncVat": 6400,
            "date": "2026-03-23",
        }))

    async def test_create_travel_cost_aligns_undated_taxi_date_to_normalized_return(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_cost",
            {
                "comments": "Taxi",
                "amountCurrencyIncVat": 600,
                "date": "2026-03-22",
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_id=555,
                last_payment_type_id=9,
                last_travel_expense_departure_date="2026-03-23",
                last_travel_expense_return_date="2026-03-24",
                last_cost_categories=[
                    {"id": 11, "displayName": "Flight"},
                    {"id": 12, "displayName": "Taxi"},
                ],
                prompt_text="Register a travel expense for Conference Bergen. The trip lasted 2 days with per diem.",
            ),
        )

        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/cost", {
            "travelExpense": {"id": 555},
            "costCategory": {"id": 12},
            "paymentType": {"id": 9},
            "comments": "Taxi",
            "amountCurrencyIncVat": 600,
            "date": "2026-03-24",
        }))

    async def test_create_travel_cost_adjusts_second_taxi_to_return_date_and_removes_redundant_rate(self):
        client = FakeTripletexClient()

        await _execute(
            client,
            "create_travel_cost",
            {
                "comments": "Táxi",
                "amountCurrencyIncVat": 350,
                "rate": 350,
                "date": "2026-03-17",
            },
            endpoint_search=None,
            ctx=EntityContext(
                last_travel_expense_id=555,
                last_payment_type_id=9,
                last_travel_expense_departure_date="2026-03-17",
                last_travel_expense_return_date="2026-03-21",
                last_cost_categories=[
                    {"id": 11, "displayName": "Flyreise"},
                    {"id": 12, "displayName": "Taxi"},
                ],
                travel_cost_count=1,
                prompt_text="Registe uma despesa de viagem para Visita cliente Trondheim. A viagem durou 5 dias com ajudas de custo.",
            ),
        )

        self.assertEqual(client.calls[-1], ("POST", "/travelExpense/cost", {
            "travelExpense": {"id": 555},
            "costCategory": {"id": 12},
            "paymentType": {"id": 9},
            "comments": "Táxi",
            "amountCurrencyIncVat": 350,
            "date": "2026-03-21",
        }))

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

    async def test_create_voucher_receipt_prefers_default_vat_over_other_legal_guess(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            account_cache={
                364015653: {
                    "id": 364015653,
                    "number": 7140,
                    "name": "Reisekostnad",
                    "vatType": {"id": 61},
                    "legalVatTypes": [{"id": 12}, {"id": 61}],
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
                "date": "2026-03-13",
                "description": "NSB kvittering - Togbillett",
                "postings": [
                    {
                        "account": {"id": 364015653},
                        "amountGross": 109.00,
                        "description": "Togbillett",
                        "vatType": {"id": 12},
                    },
                    {
                        "account": {"id": 364015350},
                        "amountGross": -109.00,
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

    async def test_create_voucher_blocks_duplicate_correction_with_guessed_bank_balancing_account(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "We have discovered errors in the general ledger for January and February 2026. "
                "Review all vouchers and find the 4 errors: a duplicate voucher (account 6860, amount 3900 NOK)."
            ),
            account_cache={
                466691059: {"id": 466691059, "number": 6860, "name": "Kontorkostnad"},
                466690878: {"id": 466690878, "number": 1920, "name": "Bank", "isBankAccount": True},
            },
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Correction voucher: reverse duplicate voucher on account 6860",
                "year": 2026,
                "postings": [
                    {"account": {"id": 466691059}, "amountGross": -3900, "amountGrossCurrency": -3900},
                    {"account": {"id": 466690878}, "amountGross": 3900, "amountGrossCurrency": 3900},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertIn("use reverse_voucher", result["error"])
        self.assertEqual(client.calls, [])

    async def test_create_voucher_blocks_wrong_amount_correction_with_guessed_bank_balancing_account(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "We have discovered errors in the general ledger for January and February 2026. "
                "Review all vouchers and find the 4 errors: an incorrect amount (account 6340, 21950 NOK posted instead of 15550 NOK)."
            ),
            account_cache={
                466691025: {"id": 466691025, "number": 6340, "name": "Leie maskiner"},
                466690878: {"id": 466690878, "number": 1920, "name": "Bank", "isBankAccount": True},
            },
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Correction voucher: correct overstated amount on account 6340",
                "year": 2026,
                "postings": [
                    {"account": {"id": 466691025}, "amountGross": -6400, "amountGrossCurrency": -6400},
                    {"account": {"id": 466690878}, "amountGross": 6400, "amountGrossCurrency": 6400},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertIn("original voucher's counterpart account", result["error"])
        self.assertEqual(client.calls, [])

    async def test_create_voucher_blocks_wrong_amount_correction_with_guessed_bank_balancing_account_for_french_ledger_prompt(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Nous avons découvert des erreurs dans le grand livre de janvier et février 2026. "
                "Vérifiez toutes les pièces et trouvez les 4 erreurs : un montant incorrect "
                "(compte 6590, 10000 NOK comptabilisé au lieu de 7200 NOK)."
            ),
            account_cache={
                466919149: {"id": 466919149, "number": 6590, "name": "Andre driftskostnader"},
                466918873: {"id": 466918873, "number": 1920, "name": "Bank", "isBankAccount": True},
            },
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Correction incorrect amount on account 6590 from 10000 to 7200",
                "year": 2026,
                "postings": [
                    {"account": {"id": 466918873}, "amountGross": 2800, "amountGrossCurrency": 2800},
                    {"account": {"id": 466919149}, "amountGross": -2800, "amountGrossCurrency": -2800},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertIn("original voucher's counterpart account", result["error"])
        self.assertEqual(client.calls, [])

    async def test_create_voucher_blocks_duplicate_correction_with_guessed_bank_balancing_account_for_nynorsk_ledger_prompt(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Me har oppdaga feil i hovudboka for januar og februar 2026. "
                "GÃ¥ gjennom alle bilag og finn dei 4 feila: eit duplikat bilag "
                "(konto 6860, belÃ¸p 2050 kr)."
            ),
            account_cache={
                466691059: {"id": 466691059, "number": 6860, "name": "Kontorkostnad"},
                466690878: {"id": 466690878, "number": 1920, "name": "Bank", "isBankAccount": True},
            },
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Korriger duplikat bilag pÃ¥ konto 6860",
                "year": 2026,
                "postings": [
                    {"account": {"id": 466691059}, "amountGross": -2050, "amountGrossCurrency": -2050},
                    {"account": {"id": 466690878}, "amountGross": 2050, "amountGrossCurrency": 2050},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertIn("use reverse_voucher", result["error"])
        self.assertEqual(client.calls, [])

    async def test_create_voucher_blocks_wrong_amount_correction_with_guessed_bank_balancing_account_for_nynorsk_ledger_prompt(self):
        client = FakeTripletexClient()
        ctx = EntityContext(
            prompt_text=(
                "Me har oppdaga feil i hovudboka for januar og februar 2026. "
                "Ga gjennom alle bilag og finn dei 4 feila: eit feil belop "
                "(konto 6590, 15100 kr bokfort i staden for 5450 kr)."
            ),
            account_cache={
                466919149: {"id": 466919149, "number": 6590, "name": "Andre driftskostnader"},
                466918873: {"id": 466918873, "number": 1920, "name": "Bank", "isBankAccount": True},
            },
        )

        result = await _execute(
            client,
            "create_voucher",
            {
                "date": "2026-03-21",
                "description": "Korriger feil belop pa konto 6590",
                "year": 2026,
                "postings": [
                    {"account": {"id": 466918873}, "amountGross": 9650, "amountGrossCurrency": 9650},
                    {"account": {"id": 466919149}, "amountGross": -9650, "amountGrossCurrency": -9650},
                ],
            },
            endpoint_search=None,
            ctx=ctx,
        )

        self.assertIn("original voucher's counterpart account", result["error"])
        self.assertEqual(client.calls, [])

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
