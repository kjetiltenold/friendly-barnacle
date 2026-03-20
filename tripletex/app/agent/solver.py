"""Main agent orchestrator — interprets task prompts and executes Tripletex API calls."""

import logging
import time
from typing import Any

from app.config import get_settings
from app.endpoint_search import EndpointSearchClient
from app.models import SolveRequest
from app.tripletex.client import TripletexClient
from app.attachments.parser import process_attachments
from app.agent.llm import create_client, chat
from app.agent.tools import dispatch_tool
from app.agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def solve_task(request: SolveRequest) -> None:
    start = time.monotonic()
    creds = request.tripletex_credentials
    tx = TripletexClient(creds.base_url, creds.session_token)
    llm = create_client()
    settings = get_settings()
    endpoint_search = EndpointSearchClient.from_settings(settings)

    try:
        # Build user message with prompt + attachments
        content = _build_user_content(request)
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        for iteration in range(settings.max_agent_iterations):
            elapsed = time.monotonic() - start
            if elapsed > settings.soft_timeout_seconds:
                logger.warning(f"Soft timeout at {elapsed:.0f}s, stopping agent")
                break

            response = await chat(llm, messages, SYSTEM_PROMPT)
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

            # Execute each tool call and collect results
            for tc in message.tool_calls:
                if tc.type != "function":
                    logger.warning(f"Skipping unsupported tool call type: {tc.type}")
                    continue

                result_str = await dispatch_tool(
                    tx,
                    tc.function.name,
                    tc.function.arguments,
                    endpoint_search=endpoint_search,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            logger.warning(f"Agent hit max iterations ({settings.max_agent_iterations})")

    finally:
        if endpoint_search is not None:
            await endpoint_search.close()
        await tx.close()


def _build_user_content(request: SolveRequest) -> str:
    """Build the initial user message content."""
    parts = []

    # Process file attachments (PDFs → text, images → described)
    if request.files:
        attachment_blocks = process_attachments(request.files)
        for block in attachment_blocks:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "image":
                parts.append("[Attached image — see file attachments]")

    parts.append(f"Complete this accounting task in Tripletex:\n\n{request.prompt}")
    return "\n\n".join(parts)
