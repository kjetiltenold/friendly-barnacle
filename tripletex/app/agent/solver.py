"""Main agent orchestrator — interprets task prompts and executes Tripletex API calls."""

import logging
import time

from app.config import get_settings
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

    try:
        # Build user message with prompt + attachments
        content = _build_user_content(request)
        messages = [{"role": "user", "content": content}]

        for iteration in range(get_settings().max_agent_iterations):
            elapsed = time.monotonic() - start
            if elapsed > get_settings().soft_timeout_seconds:
                logger.warning(f"Soft timeout at {elapsed:.0f}s, stopping agent")
                break

            response = await chat(llm, messages, SYSTEM_PROMPT)

            # Check if the model wants to use tools
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls:
                # Model is done (text-only response)
                logger.info(
                    f"Agent done after {iteration + 1} iterations, "
                    f"{tx.call_count} API calls, {tx.error_count} errors"
                )
                break

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for tc in tool_calls:
                result_str = await dispatch_tool(tx, tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning(f"Agent hit max iterations ({get_settings().max_agent_iterations})")

    finally:
        await tx.close()


def _build_user_content(request: SolveRequest) -> list[dict]:
    """Build the initial user message content blocks."""
    content: list[dict] = []

    # Process file attachments first (PDFs, images)
    if request.files:
        attachment_blocks = process_attachments(request.files)
        content.extend(attachment_blocks)

    # Add the task prompt
    content.append({
        "type": "text",
        "text": f"Complete this accounting task in Tripletex:\n\n{request.prompt}",
    })

    return content
