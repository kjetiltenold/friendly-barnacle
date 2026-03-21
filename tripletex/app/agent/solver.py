"""Main agent orchestrator — interprets task prompts and executes Tripletex API calls."""

import datetime
import json
import logging
import time
import unicodedata
from typing import Any

from app.config import get_settings
from app.endpoint_search import EndpointSearchClient
from app.models import SolveRequest
from app.tripletex.client import TripletexClient
from app.attachments.parser import process_attachments
from app.agent.llm import create_client, chat
from app.agent.tools import dispatch_tool, EntityContext
from app.agent.prompts import get_system_prompt

logger = logging.getLogger(__name__)
TERMINAL_PROXY_TOKEN_MARKER = "invalid or expired proxy token"


class TerminalTripletexProxyTokenError(RuntimeError):
    """Raised when the submission-specific Tripletex proxy token is already invalid."""


def _is_terminal_tripletex_proxy_token_error_message(message: str | None) -> bool:
    return TERMINAL_PROXY_TOKEN_MARKER in str(message or "").lower()


def _normalize_text(value: str | None) -> str:
    lowered = (value or "").lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _prompt_likely_requires_writes(prompt: str) -> bool:
    normalized = _normalize_text(prompt)
    write_markers = (
        "create",
        "opprett",
        "registrer",
        "registe",
        "enregistrez",
        "registre",
        "register",
        "update",
        "oppdater",
        "delete",
        "slett",
        "reverse",
        "book ",
        "buchen",
        "crie",
        "crear",
        "crea",
        "cree",
        "créer",
        "criar",
        "creer",
        "creez",
        "erstell",
    )
    return any(marker in normalized for marker in write_markers) or _prompt_likely_requires_invoice_payment(prompt)


def _prompt_likely_requires_invoice_payment(prompt: str) -> bool:
    normalized = _normalize_text(prompt)
    invoice_markers = (
        "invoice",
        "fatura",
        "factura",
        "facture",
        "faktura",
        "rechnung",
    )
    payment_markers = (
        "payment",
        "pay ",
        "pagamento",
        "pagar",
        "betaling",
        "betale",
        "paiement",
        "payer",
        "zahlung",
        "zahlungen",
        "teilzahlung",
        "teilzahlungen",
        "zahlung",
        "bezahlen",
        "registe o pagamento",
        "registrer betaling",
        "register payment",
        "registrez le paiement",
    )
    return any(marker in normalized for marker in invoice_markers) and any(
        marker in normalized for marker in payment_markers
    )


def _prompt_likely_requires_bank_reconciliation_payments(prompt: str) -> bool:
    normalized = _normalize_text(prompt)
    statement_markers = (
        "bank statement",
        "kontoauszug",
        "bankutskrift",
        "csv",
        "reconcile",
        "avstemm",
        "gleichen sie",
        "gleichen sie den kontoauszug",
        "abgleichen",
    )
    customer_markers = (
        "customer invoice",
        "kundenrechnung",
        "kundenrechnungen",
        "incoming payment",
        "eingehende zahlung",
        "eingehende zahlungen",
        "incoming payments",
    )
    supplier_markers = (
        "supplier invoice",
        "lieferantenrechnung",
        "lieferantenrechnungen",
        "outgoing payment",
        "ausgehende zahlung",
        "ausgehende zahlungen",
        "outgoing payments",
    )
    return (
        any(marker in normalized for marker in statement_markers)
        and any(marker in normalized for marker in customer_markers)
        and any(marker in normalized for marker in supplier_markers)
    )


def _prompt_likely_requires_contract_onboarding_completion(prompt: str) -> bool:
    normalized = _normalize_text(prompt)
    onboarding_markers = (
        "arbeidskontrakt",
        "employment contract",
        "offer letter",
        "offerletter",
        "job offer",
        "joboffer",
        "tilbudsbrev",
        "tilbodsbrev",
        "onboarding",
        "contrato de trabajo",
        "carta de oferta",
        "contrat de travail",
        "contrato de trabalho",
    )
    return any(marker in normalized for marker in onboarding_markers)


def _should_retry_text_only_response(
    assistant_text: str,
    prompt: str,
    write_call_count: int,
    reminder_count: int,
    ctx: EntityContext | None = None,
) -> bool:
    if reminder_count >= 2:
        return False
    normalized = assistant_text.strip().upper()
    if normalized != "DONE":
        return True
    if not _prompt_likely_requires_writes(prompt):
        return False
    if write_call_count == 0:
        return True
    if _prompt_likely_requires_bank_reconciliation_payments(prompt):
        return (
            getattr(ctx, "customer_invoice_payment_action_count", 0) == 0
            or getattr(ctx, "supplier_invoice_payment_action_count", 0) == 0
        )
    if _prompt_likely_requires_invoice_payment(prompt):
        return getattr(ctx, "invoice_payment_action_count", 0) == 0
    if _prompt_likely_requires_contract_onboarding_completion(prompt):
        missing_employment_details = getattr(ctx, "last_employment_details_id", None) is None
        missing_standard_time = getattr(ctx, "last_standard_time_id", None) is None
        missing_occupation_code = not getattr(
            ctx,
            "last_employment_details_had_occupation_code",
            False,
        )
        if missing_employment_details or missing_standard_time or missing_occupation_code:
            if missing_occupation_code:
                logger.info("Onboarding task missing occupation code in employment details; continuing before DONE")
            elif missing_employment_details:
                logger.info("Onboarding task missing employment details; continuing before DONE")
            else:
                logger.info("Onboarding task missing standard time; continuing before DONE")
            return True
    return False


def _build_incomplete_task_reminder(prompt: str, ctx: EntityContext) -> str:
    if isinstance(ctx.last_top_expense_analysis, dict):
        account_names = [
            str((item.get("account") or {}).get("name") or "").strip()
            for item in (ctx.last_top_expense_analysis.get("topAccounts") or [])
            if isinstance(item, dict)
        ]
        account_names = [name for name in account_names if name]
        accounts_text = ", ".join(account_names[:3])
        if accounts_text:
            return (
                "The task is not complete yet. Do not call find_top_expense_account_increases again. "
                f"Use the existing topAccounts result ({accounts_text}) and now execute the required write tools: "
                "create_project with isInternal=true for each account name, create_activity with the same name, "
                "and create_project_activity to link each activity to its project. Reply only with DONE when the writes are finished."
            )
    if _prompt_likely_requires_bank_reconciliation_payments(prompt):
        customer_payment_hint = (
            f"Use customer paymentType id={ctx.last_payment_type_id}. "
            if ctx.last_payment_type_id is not None
            else "Use GET /invoice/paymentType for incoming customer payments. "
        )
        return (
            "The task is not complete yet. Reconcile the attached bank-statement rows by executing payment writes, not just invoice lookups. "
            "Register incoming payments on matching customer invoices with PUT /invoice/{invoice_id}/:payment, "
            "and register outgoing payments on matching supplier invoices with PUT /supplierInvoice/{invoice_id}/:addPayment. "
            + customer_payment_hint
            + "Handle partial payments by paying only the transaction amount from each attached row. "
            "Reply only with DONE when both customer and supplier payment registrations are finished."
        )
    if _prompt_likely_requires_invoice_payment(prompt):
        payment_type_hint = (
            f"Use paymentType id={ctx.last_payment_type_id}. "
            if ctx.last_payment_type_id is not None
            else "Use GET /invoice/paymentType if you still need the payment type. "
        )
        invoice_hint = (
            f"Use the matched invoice id={ctx.last_invoice_id}. "
            if ctx.last_invoice_id is not None
            else "Find the matching open invoice first by customer and outstanding amount/description. "
        )
        return (
            "The task is not complete yet. Do not stop after customer or invoice lookup. "
            + invoice_hint
            + payment_type_hint
            + "Register the payment with PUT /invoice/{invoice_id}/:payment using paymentDate, paymentTypeId, and paidAmount. "
            "Reply only with DONE when the payment registration is finished."
        )
    if _prompt_likely_requires_contract_onboarding_completion(prompt):
        missing_actions: list[str] = []
        if ctx.last_employment_details_id is None:
            missing_actions.append(
                "call create_employment_details with the attachment's salary, FTE, employment type, workingHoursScheme, and occupationCodeCode or occupationCodeName"
            )
        elif not ctx.last_employment_details_had_occupation_code:
            missing_actions.append(
                "update or recreate create_employment_details so it includes occupationCodeCode or occupationCodeName copied literally from the attachment"
            )
        if ctx.last_standard_time_id is None:
            missing_actions.append(
                "call create_standard_time with the attachment's literal standard working hours"
            )
        missing_text = " Then ".join(missing_actions) if missing_actions else (
            "re-inspect the attachment and complete the remaining onboarding writes"
        )
        return (
            "The onboarding task is not complete yet. Re-inspect the attached contract or offer letter. "
            + missing_text
            + ". Reply only with DONE when the employee is created, employment details are written, standard time is configured, and employment details include an occupation code."
        )
    return (
        "The task is not complete yet. Execute all requested Tripletex create, update, delete, "
        "or posting actions before stopping. Reply only with DONE when the requested actions are finished."
    )


async def solve_task(request: SolveRequest) -> None:
    start = time.monotonic()
    creds = request.tripletex_credentials
    tx = TripletexClient(creds.base_url, creds.session_token)
    llm = create_client()
    settings = get_settings()
    endpoint_search = EndpointSearchClient.from_settings(settings)

    try:
        # Build system prompt with today's date
        today = datetime.date.today().isoformat()
        system_prompt = get_system_prompt(today)
        ctx = EntityContext()
        ctx.prompt_text = request.prompt
        completion_reminder_count = 0

        await _prime_context(tx, ctx)

        # Build user message with prompt + attachments
        content = _build_user_content(request)
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        for iteration in range(settings.max_agent_iterations):
            elapsed = time.monotonic() - start
            if elapsed > settings.soft_timeout_seconds:
                logger.warning(f"Soft timeout at {elapsed:.0f}s, stopping agent")
                break

            try:
                response = await chat(llm, messages, system_prompt)
            except Exception as e:
                error_msg = str(e).lower()
                if "invalid_prompt" in error_msg or "usage policy" in error_msg:
                    logger.warning("OpenAI content policy flag — trimming context and retrying")
                    # Keep only the original user message + last 4 messages
                    if len(messages) > 5:
                        messages = [messages[0]] + messages[-4:]
                    try:
                        response = await chat(llm, messages, system_prompt)
                    except Exception:
                        logger.error("Content policy flag persists after trim — stopping")
                        break
                else:
                    raise
            message = response.choices[0].message

            # Check if the model wants to use tools
            if not message.tool_calls:
                assistant_text = (message.content or "").strip()
                if _should_retry_text_only_response(
                    assistant_text,
                    request.prompt,
                    tx.write_call_count,
                    completion_reminder_count,
                    ctx,
                ):
                    completion_reminder_count += 1
                    logger.info("Model stopped before completing requested actions; nudging to continue")
                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append({
                        "role": "user",
                        "content": _build_incomplete_task_reminder(request.prompt, ctx),
                    })
                    continue
                # Model is done (text-only response)
                logger.info(
                    f"Agent done after {iteration + 1} iterations, "
                    f"{tx.call_count} API calls, {tx.error_count} errors"
                )
                break

            # Add assistant response to conversation
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                    if tc.type == "function"
                ],
            })

            # Execute tool calls — in parallel when multiple are requested
            function_calls = [tc for tc in message.tool_calls if tc.type == "function"]
            messages.extend(await _execute_tool_calls(tx, function_calls, endpoint_search, ctx))

            # Compress older tool results to save context window
            _compress_messages(messages)
        else:
            logger.warning(f"Agent hit max iterations ({settings.max_agent_iterations})")
    except TerminalTripletexProxyTokenError as e:
        logger.warning(f"Agent stopping due to invalid or expired Tripletex proxy token: {e}")

    finally:
        if endpoint_search is not None:
            await endpoint_search.close()
        await tx.close()


def _compress_messages(messages: list[dict], keep_recent: int = 6) -> None:
    """Compress old tool result messages to save context window.

    Keeps messages[0] (original user message) and the last `keep_recent`
    messages intact. Older write-result payloads are summarized, while
    lookup/search payloads remain intact because the model may still need
    their metadata later.
    """
    if len(messages) <= keep_recent + 1:
        return
    cutoff = len(messages) - keep_recent
    for i in range(1, cutoff):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if len(content) <= 200:
            continue
        try:
            data = json.loads(content)
            if "error" in data:
                continue
            if "values" in data and isinstance(data["values"], list):
                continue
            if "value" in data and isinstance(data["value"], dict):
                msg["content"] = json.dumps({"value": _summarize_value(data["value"])}, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass


async def _prime_context(tx: TripletexClient, ctx: EntityContext) -> None:
    """Seed context with cheap read-only lookups only."""
    try:
        result = await tx.get("/department", params={"fields": "id,name", "count": 1})
        values = result.get("values", [])
        if values:
            ctx.last_department_id = values[0].get("id")
            ctx.last_department_id_prefetched = True
    except Exception as e:
        if _is_terminal_tripletex_proxy_token_error_message(str(e)):
            logger.warning("Stopping before agent loop due to invalid or expired Tripletex proxy token during department prefetch")
            raise TerminalTripletexProxyTokenError(str(e)) from e
        logger.info(f"Department prefetch skipped: {e}")


async def _execute_tool_calls(
    tx: TripletexClient,
    function_calls: list[Any],
    endpoint_search: EndpointSearchClient | None,
    ctx: EntityContext,
) -> list[dict[str, str]]:
    """Execute tool calls in order and return tool-role messages."""
    tool_messages: list[dict[str, str]] = []
    for tc in function_calls:
        result_str = await dispatch_tool(
            tx,
            tc.function.name,
            tc.function.arguments,
            endpoint_search=endpoint_search,
            ctx=ctx,
        )
        tool_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
        try:
            result = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            result = None
        if isinstance(result, dict) and _is_terminal_tripletex_proxy_token_error_message(result.get("error")):
            logger.warning("Stopping tool execution due to invalid or expired Tripletex proxy token")
            raise TerminalTripletexProxyTokenError(str(result.get("error")))
    return tool_messages


def _summarize_value(value: dict, depth: int = 0) -> dict:
    """Keep a compact but still useful shape for older write results."""
    summary: dict[str, Any] = {}
    scalar_keys = {
        "id",
        "name",
        "displayName",
        "firstName",
        "lastName",
        "email",
        "number",
        "organizationNumber",
        "invoiceNumber",
        "title",
        "date",
        "startDate",
        "endDate",
        "userType",
        "employmentType",
        "employmentForm",
        "remunerationType",
        "workingHoursScheme",
        "annualSalary",
        "percentageOfFullTimeEquivalent",
        "hoursPerDay",
        "hours",
        "amount",
        "amountOutstanding",
    }
    for key in scalar_keys:
        if key in value and value[key] not in (None, "", [], {}):
            summary[key] = value[key]
    if depth >= 1:
        return summary
    for key, nested in value.items():
        if key in summary or nested in (None, "", [], {}):
            continue
        if isinstance(nested, dict):
            nested_summary = _summarize_value(nested, depth + 1)
            if nested_summary:
                summary[key] = nested_summary
        elif isinstance(nested, list):
            condensed_items = []
            for item in nested[:3]:
                if isinstance(item, dict):
                    item_summary = _summarize_value(item, depth + 1)
                    if item_summary:
                        condensed_items.append(item_summary)
                elif item not in (None, "", [], {}):
                    condensed_items.append(item)
            if condensed_items:
                summary[key] = condensed_items
    return summary


def _build_user_content(request: SolveRequest) -> str | list[dict]:
    """Build the initial user message content.

    Returns a plain string when there are no images, or a list of
    OpenAI-format content blocks when images are present (multimodal).
    """
    text_parts: list[str] = []
    multimodal_blocks: list[dict[str, Any]] = []
    has_images = False

    if request.files:
        attachment_guidance = (
            "[Attachment handling]\n"
            "Treat attached files as the source of truth for exact names, dates, invoice numbers, and amounts. "
            "Preserve European decimal separators when converting amounts: 109,00 means 109.00 and 51 312,50 means 51312.50. "
            "Do not translate literal supplier names, invoice titles, or line descriptions from attached invoices into another language. "
            "If extracted text conflicts with an attached image, trust the image. "
            "For single-page PDFs such as receipts, contracts, and offer letters, inspect the image first because OCR may flatten layout or structured fields. "
            "For contracts and offer letters, preserve literal daily or weekly working-hours figures from the attachment instead of deriving standard hours from FTE when explicit hours are shown."
        )
        text_parts.append(attachment_guidance)
        multimodal_blocks.append({"type": "text", "text": attachment_guidance})
        attachment_blocks = process_attachments(request.files)
        for block in attachment_blocks:
            if block["type"] == "text":
                text_parts.append(block["text"])
                multimodal_blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                # Convert from Anthropic image format to OpenAI image_url format
                source = block["source"]
                mime = source["media_type"]
                data = source["data"]
                has_images = True
                multimodal_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })

    task_text = f"Complete this accounting task in Tripletex:\n\n{request.prompt}"
    text_parts.append(task_text)
    full_text = "\n\n".join(text_parts)

    if not has_images:
        return full_text

    multimodal_blocks.append({"type": "text", "text": task_text})
    return multimodal_blocks
