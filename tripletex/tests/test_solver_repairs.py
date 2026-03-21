import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.solver import (
    _build_user_content,
    _compress_messages,
    _execute_tool_calls,
    _prime_context,
    _should_retry_text_only_response,
)
from app.agent.prompts import get_system_prompt
from app.agent.tools import EntityContext
from app.models import FileAttachment, SolveRequest, TripletexCredentials


class FakeTripletexClient:
    def __init__(self, get_responses=None):
        self.get_responses = get_responses or {}
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        key = (path, tuple(sorted((params or {}).items())))
        return self.get_responses.get(key, {"fullResultSize": 0, "values": []})


def _tool_call(tool_call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class SolverRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_prime_context_prefetches_department_read_only(self):
        client = FakeTripletexClient(
            get_responses={
                (
                    "/department",
                    (("count", 1), ("fields", "id,name")),
                ): {"fullResultSize": 1, "values": [{"id": 55, "name": "Salg"}]},
            }
        )
        ctx = EntityContext()

        await _prime_context(client, ctx)

        self.assertEqual(ctx.last_department_id, 55)
        self.assertEqual(
            client.calls,
            [("GET", "/department", {"fields": "id,name", "count": 1})],
        )

    async def test_execute_tool_calls_runs_in_order_with_shared_context(self):
        ctx = EntityContext()
        observed_customer_ids = []

        async def fake_dispatch_tool(tx, name, args_json, endpoint_search=None, ctx=None):
            if name == "create_customer":
                ctx.last_customer_id = 123
                return json.dumps({"value": {"id": 123}})
            if name == "create_order":
                observed_customer_ids.append(ctx.last_customer_id)
                return json.dumps({"value": {"id": 456, "customer": {"id": ctx.last_customer_id}}})
            raise AssertionError(f"Unexpected tool {name}")

        tool_calls = [
            _tool_call("tc1", "create_customer", {"name": "Acme"}),
            _tool_call("tc2", "create_order", {"orderDate": "2026-03-21"}),
        ]

        with patch("app.agent.solver.dispatch_tool", side_effect=fake_dispatch_tool):
            messages = await _execute_tool_calls(FakeTripletexClient(), tool_calls, None, ctx)

        self.assertEqual(observed_customer_ids, [123])
        self.assertEqual(
            messages,
            [
                {"role": "tool", "tool_call_id": "tc1", "content": json.dumps({"value": {"id": 123}})},
                {"role": "tool", "tool_call_id": "tc2", "content": json.dumps({"value": {"id": 456, "customer": {"id": 123}}})},
            ],
        )

    def test_compress_messages_keeps_lookup_payloads(self):
        lookup_payload = json.dumps(
            {
                "values": [
                    {
                        "id": 10,
                        "number": 7350,
                        "name": "Representasjon",
                        "vatLocked": True,
                        "requiresDepartment": False,
                        "isApplicableForSupplierInvoice": True,
                    }
                ],
                "fullResultSize": 1,
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "t1", "content": lookup_payload + (" " * 220)},
            {"role": "assistant", "content": "next"},
        ]

        _compress_messages(messages, keep_recent=1)

        self.assertIn('"vatLocked": true', messages[1]["content"])
        self.assertIn('"requiresDepartment": false', messages[1]["content"])

    def test_compress_messages_preserves_useful_nested_ids(self):
        create_payload = json.dumps(
            {
                "value": {
                    "id": 18619016,
                    "firstName": "Marit",
                    "lastName": "Lunde",
                    "email": "marit.lunde@example.org",
                    "department": {"id": 927020, "name": "Utvikling", "displayName": "Utvikling"},
                    "employments": [{"id": 2813211, "startDate": "2026-07-25"}],
                    "comments": "x" * 300,
                }
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "t1", "content": create_payload},
            {"role": "assistant", "content": "next"},
        ]

        _compress_messages(messages, keep_recent=1)

        compressed = json.loads(messages[1]["content"])
        self.assertEqual(compressed["value"]["id"], 18619016)
        self.assertEqual(compressed["value"]["department"]["id"], 927020)
        self.assertEqual(compressed["value"]["employments"][0]["id"], 2813211)

    def test_build_user_content_adds_attachment_source_of_truth_note(self):
        request = SolveRequest(
            prompt="Post this receipt correctly.",
            files=[
                FileAttachment(
                    filename="receipt.txt",
                    mime_type="text/plain",
                    content_base64="VG9nYmlsbGV0dApOU0IKMTA5LDAwCg==",
                )
            ],
            tripletex_credentials=TripletexCredentials(
                base_url="https://example.invalid/v2",
                session_token="token",
            ),
        )

        content = _build_user_content(request)

        self.assertIsInstance(content, str)
        self.assertIn("Treat attached files as the source of truth", content)
        self.assertIn("109,00 means 109.00", content)

    def test_should_retry_text_only_response_when_model_stops_without_done(self):
        self.assertTrue(
            _should_retry_text_only_response(
                "Here are the top three accounts.",
                "Analyze the increases and create a project for each.",
                0,
                0,
            )
        )

    def test_should_retry_text_only_response_when_done_but_no_requested_writes_happened(self):
        self.assertTrue(
            _should_retry_text_only_response(
                "DONE",
                "Crie um projeto interno para cada conta.",
                0,
                0,
            )
        )

    def test_should_not_retry_text_only_response_when_done_after_writes(self):
        self.assertFalse(
            _should_retry_text_only_response(
                "DONE",
                "Create a project and activity.",
                2,
                0,
            )
        )

    def test_should_not_retry_text_only_response_after_two_reminders(self):
        self.assertFalse(
            _should_retry_text_only_response(
                "DONE",
                "Create a project and activity.",
                0,
                2,
            )
        )

    def test_system_prompt_includes_vat_correction_guidance(self):
        prompt = get_system_prompt("2026-03-21")

        self.assertIn("debiting the input VAT account such as `2710` and crediting the original expense account", prompt)
        self.assertIn("Do not credit bank `1920`", prompt)

    def test_system_prompt_includes_foreign_currency_invoice_guidance(self):
        prompt = get_system_prompt("2026-03-21")

        self.assertIn("foreign-currency invoice", prompt)
        self.assertIn("do not invent 25 percent VAT", prompt)
        self.assertIn("debit the exchange-loss account such as `8160` and credit accounts receivable `1500`", prompt)

    def test_system_prompt_includes_month_end_closing_guidance(self):
        prompt = get_system_prompt("2026-03-21")

        self.assertIn("Month-end closing", prompt)
        self.assertIn("Do not mistake an amount like 8300 NOK for account `8300`", prompt)
        self.assertIn("Post accrual reversal, depreciation, and salary accrual as separate vouchers", prompt)

    def test_system_prompt_includes_project_budget_and_timesheet_guidance(self):
        prompt = get_system_prompt("2026-03-21")

        self.assertIn("A project budget is not the same as a fixed-price project", prompt)
        self.assertIn("typically 7.5 or 8 hours per day, not 24-hour days", prompt)

    def test_system_prompt_includes_supplier_invoice_attachment_and_software_account_guidance(self):
        prompt = get_system_prompt("2026-03-21")

        self.assertIn("preserve the literal supplier name, organization number, invoice number, invoice date, and line description", prompt)
        self.assertIn("Prefer account `6420`", prompt)


if __name__ == "__main__":
    unittest.main()
