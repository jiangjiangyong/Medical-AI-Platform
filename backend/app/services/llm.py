from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import settings


class DeepSeekService:
    def __init__(self) -> None:
        self.enabled = bool(settings.deepseek_api_key)
        self.client = (
            AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=45.0,
                max_retries=1,
            )
            if self.enabled
            else None
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if not self.client:
            return fallback, "template-fallback"
        try:
            response = await self.client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1800,
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json(content)
            if not parsed:
                return fallback, "deepseek-empty-fallback"
            return parsed, settings.deepseek_model
        except Exception:
            return fallback, "deepseek-error-fallback"

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | None:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

