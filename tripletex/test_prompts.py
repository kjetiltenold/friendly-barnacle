"""Local test harness — sends example prompts and logs agent tool calls."""

import asyncio
import datetime
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s", stream=sys.stdout)

from app.agent.prompts import get_system_prompt
from app.agent.llm import create_client, chat
from app.agent.tools import get_tool_definitions


# Realistic fake data matching actual Tripletex sandbox responses
FAKE_VAT_TYPES = {
    "fullResultSize": 6,
    "values": [
        {"id": 3, "number": 3, "name": "Høy sats", "percentage": 25.0},
        {"id": 5, "number": 5, "name": "Middels sats", "percentage": 15.0},
        {"id": 6, "number": 6, "name": "Lav sats", "percentage": 12.0},
        {"id": 7, "number": 7, "name": "Ingen mva", "percentage": 0.0},
        {"id": 50, "number": 50, "name": "Inngående høy sats", "percentage": 25.0},
        {"id": 52, "number": 52, "name": "Inngående middels sats", "percentage": 15.0},
    ],
}

FAKE_PAYMENT_TYPES = {
    "fullResultSize": 2,
    "values": [
        {"id": 26177580, "description": "Bankoverføring"},
        {"id": 26177581, "description": "Kontant"},
    ],
}

FAKE_SALARY_TYPES = {
    "fullResultSize": 3,
    "values": [
        {"id": 48759166, "number": 2000, "name": "Fastlønn"},
        {"id": 48759170, "number": 2010, "name": "Timelønn"},
        {"id": 48759219, "number": 3000, "name": "Bonus"},
    ],
}

FAKE_ACTIVITIES = {
    "fullResultSize": 3,
    "values": [
        {"id": 5080520, "name": "Analyse"},
        {"id": 5080521, "name": "Design"},
        {"id": 5080522, "name": "Testing"},
    ],
}

FAKE_RATE_CATEGORIES = {
    "fullResultSize": 2,
    "values": [
        {"id": 740, "name": "Innland med overnatting"},
        {"id": 741, "name": "Innland uten overnatting"},
    ],
}

FAKE_COST_CATEGORIES = {
    "fullResultSize": 3,
    "values": [
        {"id": 26456260, "name": "Flybillett", "amountType": "AMOUNT"},
        {"id": 26456261, "name": "Taxi", "amountType": "AMOUNT"},
        {"id": 26456262, "name": "Hotell", "amountType": "AMOUNT"},
    ],
}

FAKE_ACCOUNTS = {
    "1920": {"id": 357101849, "number": 1920, "name": "Bank", "isBankAccount": True},
    "2400": {"id": 354552128, "number": 2400, "name": "Leverandørgjeld"},
    "6300": {"id": 354552300, "number": 6300, "name": "Leie av lokaler"},
    "6540": {"id": 354552360, "number": 6540, "name": "Kontorrekvisita"},
    "6860": {"id": 354552400, "number": 6860, "name": "Kontortjenester"},
    "7000": {"id": 354552500, "number": 7000, "name": "Lønnskostnad"},
    "7100": {"id": 354552510, "number": 7100, "name": "Arbeidsgiveravgift"},
    "7300": {"id": 354552530, "number": 7300, "name": "Kontortjenester"},
}

FAKE_HOURLY_RATES = {
    "fullResultSize": 1,
    "values": [
        {"id": 11065983, "project": {"id": 401935949}, "startDate": "2026-01-01",
         "hourlyRateModel": "TYPE_FIXED_HOURLY_RATE", "fixedRate": 0, "showInProjectOrder": False},
    ],
}

# Counter for unique IDs
_next_id = [100000000]

def _new_id():
    _next_id[0] += 1
    return _next_id[0]


def _build_fake_response(name: str, args: dict, _legacy_id: int) -> dict:
    """Build a realistic fake Tripletex response based on the tool call."""

    if name == "create_employee":
        eid = _new_id()
        return {"value": {"id": eid, "firstName": args.get("firstName", ""), "lastName": args.get("lastName", ""),
                          "email": args.get("email", ""), "userType": "STANDARD",
                          "employments": [{"startDate": args.get("startDate", "2026-01-01")}]}}

    if name == "create_customer":
        cid = _new_id()
        return {"value": {"id": cid, "name": args.get("name", ""), "organizationNumber": args.get("organizationNumber", ""),
                          "email": args.get("email", ""), "isCustomer": True, "isSupplier": args.get("isSupplier", False),
                          "supplierNumber": 20001 if args.get("isSupplier") else 0, "customerNumber": 10001}}

    if name == "create_product":
        pid = _new_id()
        return {"value": {"id": pid, "name": args.get("name", ""), "number": args.get("number", str(pid)),
                          "priceExcludingVatCurrency": args.get("priceExcludingVatCurrency", 0)}}

    if name == "create_order":
        oid = _new_id()
        return {"value": {"id": oid, "customer": args.get("customer"), "orderDate": args.get("orderDate", ""),
                          "orderLines": args.get("orderLines", [])}}

    if name == "create_invoice":
        iid = _new_id()
        return {"value": {"id": iid, "invoiceDate": args.get("invoiceDate", ""), "invoiceDueDate": args.get("invoiceDueDate", ""),
                          "amount": 50000, "amountOutstanding": 50000}}

    if name == "create_project":
        pid = _new_id()
        return {"value": {"id": pid, "name": args.get("name", ""), "number": args.get("number", "1"),
                          "projectManager": args.get("projectManager"), "customer": args.get("customer")}}

    if name == "create_department":
        did = _new_id()
        return {"value": {"id": did, "name": args.get("name", ""), "departmentNumber": args.get("departmentNumber", "1")}}

    if name == "create_travel_expense":
        tid = _new_id()
        return {"value": {"id": tid, "title": args.get("title", ""), "employee": args.get("employee")}}

    if name == "create_per_diem_compensation":
        return {"value": {"id": _new_id(), "travelExpense": args.get("travelExpense")}}

    if name == "create_travel_cost":
        return {"value": {"id": _new_id(), "travelExpense": args.get("travelExpense")}}

    if name == "create_project_activity":
        return {"value": {"id": _new_id(), "project": args.get("project"), "activity": args.get("activity")}}

    if name == "create_timesheet_entry":
        return {"value": {"id": _new_id(), "hours": args.get("hours", 0)}}

    if name == "update_project_hourly_rate":
        return {"value": {"id": args.get("hourly_rate_id", _new_id()), "fixedRate": args.get("fixedRate", 0)}}

    if name == "create_accounting_dimension_name":
        did = _new_id()
        return {"value": {"id": did, "dimensionName": args.get("dimensionName", ""), "dimensionIndex": 1}}

    if name == "create_accounting_dimension_value":
        vid = _new_id()
        return {"value": {"id": vid, "displayName": args.get("displayName", ""), "dimensionIndex": args.get("dimensionIndex", 1)}}

    if name == "create_voucher":
        return {"value": {"id": _new_id(), "date": args.get("date", ""), "number": 1}}

    if name == "create_salary_transaction":
        return {"value": {"id": _new_id()}}

    if name == "search_entity":
        etype = args.get("entity_type", "")
        if etype == "employee":
            return {"fullResultSize": 1, "values": [
                {"id": 18179406, "firstName": "Charles", "lastName": "Harris", "email": "charles.harris@example.org"}
            ]}
        if etype == "customer":
            return {"fullResultSize": 1, "values": [
                {"id": 107823243, "name": "Greenfield Ltd", "organizationNumber": "918318070"}
            ]}
        if etype == "project":
            return {"fullResultSize": 1, "values": [
                {"id": 401935949, "name": "Plattformintegrasjon", "number": "1", "projectManager": {"id": 18187428}}
            ]}
        if etype == "invoice":
            return {"fullResultSize": 1, "values": [
                {"id": 2147490974, "invoiceNumber": 10001, "amount": 50312.5, "amountOutstanding": 50312.5,
                 "customer": {"id": 107823243}, "voucher": {"id": 999001}}
            ]}
        return {"fullResultSize": 0, "values": []}

    if name == "get_entity":
        return {"value": {"id": args.get("entity_id", 1), "name": "Entity"}}

    if name == "delete_entity":
        return {}

    if name == "find_tripletex_endpoints":
        return {"results": [
            {"path": "/ledger/vatType", "method": "GET", "description": "Get VAT types"},
            {"path": "/invoice/paymentType", "method": "GET", "description": "Get payment types"},
            {"path": "/invoice/{id}/:payment", "method": "PUT", "description": "Register payment on invoice"},
            {"path": "/invoice/{id}/:createCreditNote", "method": "PUT", "description": "Create credit note"},
            {"path": "/travelExpense/rateCategory", "method": "GET", "description": "Get per diem rate categories"},
        ]}

    if name == "tripletex_api_call":
        method = args.get("method", "GET")
        path = args.get("path", "")
        # Strip query params for matching
        clean_path = path.split("?")[0]

        if "vatType" in path:
            return FAKE_VAT_TYPES
        if "paymentType" in path:
            return FAKE_PAYMENT_TYPES
        if "salary/type" in path:
            return FAKE_SALARY_TYPES
        if "/activity" in path and "projectActivity" not in path and "timesheet" not in path:
            return FAKE_ACTIVITIES
        if "rateCategory" in path:
            return FAKE_RATE_CATEGORIES
        if "costCategory" in path:
            return FAKE_COST_CATEGORIES
        if "hourlyRates" in path and method == "GET":
            return FAKE_HOURLY_RATES
        if "accountingDimensionName" in path and method == "POST":
            did = _new_id()
            body = args.get("body", {})
            return {"value": {"id": did, "dimensionName": body.get("dimensionName", ""), "dimensionIndex": 1}}
        if "accountingDimensionValue" in path and method == "POST":
            vid = _new_id()
            body = args.get("body", {})
            return {"value": {"id": vid, "displayName": body.get("displayName", ""), "dimensionIndex": body.get("dimensionIndex", 1)}}
        if "/ledger/account" in path:
            # Try to match account number from query params
            import re
            num_match = re.search(r'number=(\d+)', path)
            if num_match:
                num = num_match.group(1)
                if num in FAKE_ACCOUNTS:
                    return {"fullResultSize": 1, "values": [FAKE_ACCOUNTS[num]]}
            # Return all accounts
            return {"fullResultSize": len(FAKE_ACCOUNTS), "values": list(FAKE_ACCOUNTS.values())}
        if "/ledger/voucher" in path and method == "POST":
            vid = _new_id()
            return {"value": {"id": vid, "date": args.get("body", {}).get("date", ""), "number": 1}}
        if "projectActivity" in path and method == "POST":
            paid = _new_id()
            return {"value": {"id": paid}}
        if "timesheet/entry" in path and method == "POST":
            tid = _new_id()
            return {"value": {"id": tid}}
        if "salary/transaction" in path and method == "POST":
            sid = _new_id()
            return {"value": {"id": sid}}
        if "perDiemCompensation" in path and method == "POST":
            pdid = _new_id()
            return {"value": {"id": pdid}}
        if "travelExpense/cost" in path and method == "POST":
            cid = _new_id()
            return {"value": {"id": cid}}
        if "/:payment" in path:
            return {"value": {"id": _new_id()}}
        if "/:createCreditNote" in path:
            return {"value": {"id": _new_id()}}
        if "/:invoice" in path:
            return {"value": {"id": _new_id()}}
        if method == "PUT":
            return {"value": {"id": _new_id()}}
        if method == "GET":
            return {"fullResultSize": 0, "values": []}
        return {"value": {"id": _new_id()}}

    return {"value": {"id": _new_id()}}


async def dry_run(prompt: str, max_turns: int = 5):
    """Run the agent loop WITHOUT making real Tripletex calls.

    Sends the prompt to the LLM and prints which tool calls it would make,
    but returns fake success responses instead of hitting Tripletex.
    """
    today = datetime.date.today().isoformat()
    system = get_system_prompt(today)
    llm = create_client()

    messages = [{"role": "user", "content": f"Complete this accounting task in Tripletex:\n\n{prompt}"}]

    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")

    for turn in range(max_turns):
        response = await chat(llm, messages, system)
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"\n[Turn {turn+1}] Agent finished. Final message: {msg.content}")
            break

        # Show what the agent wants to do
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls if tc.type == "function"
            ],
        })

        for tc in msg.tool_calls:
            if tc.type != "function":
                continue
            args = json.loads(tc.function.arguments)
            print(f"\n[Turn {turn+1}] TOOL CALL: {tc.function.name}")
            print(f"  Args: {json.dumps(args, indent=2, ensure_ascii=False)}")

            # Return realistic fake responses matching real Tripletex data
            fake_id = 1000 + turn * 10 + list(msg.tool_calls).index(tc)
            name = tc.function.name
            fake_response = _build_fake_response(name, args, fake_id)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(fake_response, ensure_ascii=False),
            })
            resp_summary = json.dumps(fake_response, ensure_ascii=False)
            if len(resp_summary) > 200:
                resp_summary = resp_summary[:200] + "..."
            print(f"  Response: {resp_summary}")

    print(f"\n{'='*60}\n")


async def main():
    # Run specific test if passed as argument, otherwise run all
    import sys
    all_examples = {
        "employee": "We have a new employee named Charles Taylor, born 21. October 1994. Please create them as an employee with email charles.taylor@example.org and start date 3. June 2026.",
        "supplier": "Registre el proveedor Dorada SL con número de organización 853166553. Correo electrónico: faktura@doradasl.no.",
        "project": 'Créez le projet "Implémentation Montagne" lié au client Montagne SARL (nº org. 989074784). Le chef de projet est Lucas Robert (lucas.robert@example.org).',
        "invoice": "Opprett og send en faktura til kunden Bergvik AS (org.nr 890733751) på 28900 kr eksklusiv MVA. Fakturaen gjelder Systemutvikling.",
        "invoice_multi": 'Créez une facture pour le client Colline SARL (nº org. 942447647) avec trois lignes de produit : Service réseau (1340) à 10500 NOK avec 25 % TVA, Stockage cloud (9754) à 11000 NOK avec 15 % TVA (alimentaire), et Session de formation (7005) à 5850 NOK avec 0 % TVA (exonéré).',
        "order_payment": "Erstellen Sie einen Auftrag für den Kunden Grünfeld GmbH (Org.-Nr. 920238882) mit den Produkten Datenberatung (5628) zu 23000 NOK und Cloud-Speicher (1573) zu 16550 NOK. Wandeln Sie den Auftrag in eine Rechnung um und registrieren Sie die vollständige Zahlung.",
        "supplier_invoice": "Wir haben die Rechnung INV-2026-6392 vom Lieferanten Silberberg GmbH (Org.-Nr. 871719500) über 6500 NOK einschließlich MwSt. erhalten. Der Betrag betrifft Bürodienstleistungen (Konto 6860). Erfassen Sie die Lieferantenrechnung mit der korrekten Vorsteuer (25 %).",
        "credit_note": 'The customer Greenfield Ltd (org no. 918318070) has complained about the invoice for "Software License" (40250 NOK excl. VAT). Issue a full credit note that reverses the entire invoice.',
        "payment": 'The customer Greenfield Ltd (org no. 918318070) has an outstanding invoice for 34450 NOK excluding VAT for "Consulting Hours". Register full payment on this invoice.',
        "travel": 'Register a travel expense for Charles Harris (charles.harris@example.org) for "Client visit Oslo". The trip lasted 5 days with per diem (daily rate 800 NOK). Expenses: flight ticket 2300 NOK and taxi 500 NOK.',
        "dimensions": 'Opprett en fri regnskapsdimensjon "Marked" med verdiene "Privat" og "Bedrift". Bokfør deretter et bilag på konto 6300 for 12650 kr, knyttet til dimensjonsverdien "Privat".',
        "salary": "Führen Sie die Gehaltsabrechnung für Mia Hoffmann (mia.hoffmann@example.org) für diesen Monat durch. Das Grundgehalt beträgt 40350 NOK. Fügen Sie einen einmaligen Bonus von 7350 NOK zum Grundgehalt hinzu.",
        "timesheet": 'Registrer 5 timer for Ingrid Nilsen (ingrid.nilsen@example.org) på aktiviteten "Analyse" i prosjektet "Plattformintegrasjon" for Bergvik AS (org.nr 989231898). Timesats: 1400 kr/t. Generer en prosjektfaktura til kunden basert på de registrerte timene.',
        "reverse_payment": 'Die Zahlung von Windkraft GmbH (Org.-Nr. 823566441) für die Rechnung "Wartung" (29500 NOK ohne MwSt.) wurde von der Bank zurückgebucht. Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt.',
    }

    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        examples = {k: all_examples[k] for k in keys if k in all_examples}
        if not examples:
            print(f"Unknown test(s): {keys}")
            print(f"Available: {', '.join(all_examples.keys())}")
            return
    else:
        examples = all_examples

    for name, prompt in examples.items():
        print(f"\n>>> TEST: {name}")
        await dry_run(prompt, max_turns=8)


if __name__ == "__main__":
    asyncio.run(main())
