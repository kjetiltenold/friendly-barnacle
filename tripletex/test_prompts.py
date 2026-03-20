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

            # Return a realistic fake response matching Tripletex format
            fake_id = 1000 + turn * 10 + list(msg.tool_calls).index(tc)
            name = tc.function.name
            if name == "create_employee":
                fake_response = {"value": {"id": fake_id, "firstName": args.get("firstName", ""), "lastName": args.get("lastName", ""), "email": args.get("email", "")}}
            elif name == "create_customer":
                fake_response = {"value": {"id": fake_id, "name": args.get("name", ""), "email": args.get("email", ""), "isCustomer": True}}
            elif name == "create_product":
                fake_response = {"value": {"id": fake_id, "name": args.get("name", "")}}
            elif name == "create_order":
                fake_response = {"value": {"id": fake_id, "customer": args.get("customer"), "orderDate": args.get("orderDate", "")}}
            elif name == "create_invoice":
                fake_response = {"value": {"id": fake_id, "invoiceDate": args.get("invoiceDate", ""), "invoiceDueDate": args.get("invoiceDueDate", "")}}
            elif name == "search_entity":
                fake_response = {"fullResultSize": 0, "values": []}
            else:
                fake_response = {"value": {"id": fake_id}}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(fake_response),
            })
            print(f"  Fake response: id={fake_id}")

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
