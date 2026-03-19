"""Claude API client for tool-use conversations."""

import anthropic

from app.config import get_settings
from app.agent.tools import TOOL_DEFINITIONS


def create_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)


async def chat(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
    system: str,
) -> anthropic.types.Message:
    return await client.messages.create(
        model=get_settings().model_name,
        max_tokens=4096,
        system=system,
        messages=messages,
        tools=TOOL_DEFINITIONS,
    )
