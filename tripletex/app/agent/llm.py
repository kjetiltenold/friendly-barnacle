"""OpenAI-compatible API client for tool-use conversations."""

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import get_settings
from app.agent.tools import get_tool_definitions


def create_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)


async def chat(
    client: AsyncOpenAI,
    messages: list[dict],
    system: str,
) -> ChatCompletion:
    full_messages = [{"role": "system", "content": system}, *messages]
    return await client.chat.completions.create(
        model=get_settings().model_name,
        max_tokens=4096,
        messages=full_messages,
        tools=get_tool_definitions(),
    )
