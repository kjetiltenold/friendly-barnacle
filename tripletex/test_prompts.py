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
    examples = [
        # Tier 1: Simple entity creation
        "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",

        # Tier 1: Create customer
        "Registrer en ny kunde med navn Acme AS, epost post@acme.no, organisasjonsnummer 987654321.",

        # Tier 2: Create invoice (multi-step)
        "Opprett en faktura til kunden Bergen Consulting AS (org.nr 912345678) for 10 timer konsulenttjenester à 1500 kr ekskl. mva. Forfallsdato er 2026-04-15.",
    ]

    for prompt in examples:
        await dry_run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
