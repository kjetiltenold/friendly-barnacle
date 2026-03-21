"""Main agent orchestrator — interprets task prompts and executes Tripletex API calls."""

import asyncio
import datetime
import json
import logging
import time
from typing import Any

from app.config import get_settings
from app.endpoint_search import EndpointSearchClient
from app.models import SolveRequest
from app.tripletex.client import TripletexClient
from app.attachments.parser import process_attachments
from app.agent.llm import create_client, chat
from app.agent.tools import dispatch_tool, EntityContext, _ensure_department, _ensure_bank_account
from app.agent.prompts import get_system_prompt

logger = logging.getLogger(__name__)


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

        # Pre-fetch department and bank account to avoid reactive 4xx errors
        dept_id = await _ensure_department(tx)
        if dept_id:
            ctx.last_department_id = dept_id
        await _ensure_bank_account(tx)

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
            if len(function_calls) > 1:
                # Run independent tool calls concurrently
                coros = [
                    dispatch_tool(tx, tc.function.name, tc.function.arguments, endpoint_search=endpoint_search, ctx=ctx)
                    for tc in function_calls
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)
                for tc, result in zip(function_calls, results):
                    result_str = json.dumps({"error": str(result)}) if isinstance(result, Exception) else result
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
            else:
                for tc in function_calls:
                    result_str = await dispatch_tool(
                        tx, tc.function.name, tc.function.arguments,
                        endpoint_search=endpoint_search, ctx=ctx,
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            # Compress older tool results to save context window
            _compress_messages(messages)
        else:
            logger.warning(f"Agent hit max iterations ({settings.max_agent_iterations})")

    finally:
        if endpoint_search is not None:
            await endpoint_search.close()
        await tx.close()


def _compress_messages(messages: list[dict], keep_recent: int = 6) -> None:
    """Compress old tool result messages to save context window.

    Keeps messages[0] (original user message) and the last `keep_recent`
    messages intact. Older tool results are summarized to just entity IDs.
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
        # Extract essential info: entity ID from value.id
        try:
            data = json.loads(content)
            if "value" in data and isinstance(data["value"], dict):
                entity_id = data["value"].get("id")
                entity_name = data["value"].get("name", data["value"].get("firstName", ""))
                msg["content"] = json.dumps({"value": {"id": entity_id, "name": entity_name}})
            elif "values" in data and isinstance(data["values"], list):
                ids = [v.get("id") for v in data["values"][:5] if isinstance(v, dict)]
                msg["content"] = json.dumps({"values": [{"id": i} for i in ids], "fullResultSize": data.get("fullResultSize", len(ids))})
            elif "error" in data:
                pass  # Keep error messages as-is
        except (json.JSONDecodeError, TypeError):
            pass


def _build_user_content(request: SolveRequest) -> str | list[dict]:
    """Build the initial user message content.

    Returns a plain string when there are no images, or a list of
    OpenAI-format content blocks when images are present (multimodal).
    """
    text_parts: list[str] = []
    image_blocks: list[dict] = []

    if request.files:
        attachment_blocks = process_attachments(request.files)
        for block in attachment_blocks:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "image":
                # Convert from Anthropic image format to OpenAI image_url format
                source = block["source"]
                mime = source["media_type"]
                data = source["data"]
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })

    text_parts.append(f"Complete this accounting task in Tripletex:\n\n{request.prompt}")
    full_text = "\n\n".join(text_parts)

    if not image_blocks:
        return full_text

    # Multimodal: text + images as content block list
    return [{"type": "text", "text": full_text}, *image_blocks]
