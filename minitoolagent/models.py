from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import httpx
from openai import OpenAI

from .config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 2.0


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatModel:
    """Thin wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(config.timeout),
                write=10.0,
                pool=10.0,
            ),
            max_retries=0,
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        stop: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        for attempt in range(1 + MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                if resp.usage:
                    self.total_prompt_tokens += resp.usage.prompt_tokens
                    self.total_completion_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, 1 + MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise

    @property
    def token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
