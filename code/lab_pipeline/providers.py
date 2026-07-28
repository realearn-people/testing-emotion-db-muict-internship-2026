"""Model-provider strategies and adapters.

Strategy: every provider implements ModelProvider.infer().
Adapter: OllamaVisionAdapter / OpenAIVisionAdapter / GeminiVisionAdapter convert our common request into each provider's own format.
Factory: ProviderFactory creates the configured provider.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ProviderResponse:
    raw_output: str
    elapsed_ms: float


class ModelProvider(Protocol):
    """Common strategy used by the worker pipeline."""

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        ...


class OllamaVisionAdapter:
    """Adapter from the common ModelProvider interface to Ollama."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        temperature: float,
        top_p: float,
        num_ctx: int,
        timeout_s: float,
        retries: int,
        retry_delay_s: float,
        semaphore: asyncio.Semaphore,
    ) -> None:
        try:
            from ollama import AsyncClient
        except ImportError as exc:
            raise RuntimeError("Python package 'ollama' is required") from exc

        self._client = AsyncClient(host=host, timeout=timeout_s)
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._num_ctx = num_ctx
        self._retries = retries
        self._retry_delay_s = retry_delay_s
        self._semaphore = semaphore

    @staticmethod
    def _encode_image(path: Path) -> str:
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        encoded_images = [self._encode_image(path) for path in image_paths]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_message,
                "images": encoded_images,
            },
        ]

        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with self._semaphore:
                    started = time.perf_counter()
                    response = await self._client.chat(
                        model=self._model,
                        messages=messages,
                        options={
                            "temperature": self._temperature,
                            "top_p": self._top_p,
                            "num_ctx": self._num_ctx,
                        },
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                return ProviderResponse(
                    raw_output=str(response["message"]["content"]),
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:  # provider-specific transport failures
                last_error = exc
                if attempt >= self._retries:
                    break
                await asyncio.sleep(self._retry_delay_s * (attempt + 1))

        raise RuntimeError(f"Ollama request failed: {last_error}")

class OpenAIVisionAdapter:
    """Adapter from the common ModelProvider interface to the OpenAI API.

    Sends images as base64 data URIs using OpenAI's chat.completions
    vision format, rather than Ollama's flat "images" list.
    num_ctx is accepted for interface parity with OllamaVisionAdapter
    but is not a meaningful OpenAI parameter, so it is ignored.
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None,
        model: str,
        temperature: float,
        top_p: float,
        num_ctx: int,
        timeout_s: float,
        retries: int,
        retry_delay_s: float,
        semaphore: asyncio.Semaphore,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Python package 'openai' is required") from exc

        if not api_key:
            raise RuntimeError(
                "An OpenAI API key is required (set OPENAI_API_KEY)."
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=host,
            timeout=timeout_s,
        )
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._retries = retries
        self._retry_delay_s = retry_delay_s
        self._semaphore = semaphore

    @staticmethod
    def _encode_image_data_uri(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "image/jpeg"
        with path.open("rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image_data_uri(path)},
            }
            for path in image_paths
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    *image_content,
                ],
            },
        ]

        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with self._semaphore:
                    started = time.perf_counter()
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=self._temperature,
                        top_p=self._top_p,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                return ProviderResponse(
                    raw_output=str(response.choices[0].message.content),
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:  # provider-specific transport failures
                last_error = exc
                if attempt >= self._retries:
                    break
                await asyncio.sleep(self._retry_delay_s * (attempt + 1))

        raise RuntimeError(f"OpenAI request failed: {last_error}")

class GeminiVisionAdapter:
    """Adapter from the common ModelProvider interface to Google Gemini.

    Sends images as inline bytes using the google-genai SDK's Part.from_bytes
    helper. num_ctx and top_p behave differently on Gemini (top_p is
    supported; num_ctx is not a Gemini concept and is ignored), kept only
    for interface parity with the other adapters.
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None,
        model: str,
        temperature: float,
        top_p: float,
        num_ctx: int,
        timeout_s: float,
        retries: int,
        retry_delay_s: float,
        semaphore: asyncio.Semaphore,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Python package 'google-genai' is required") from exc

        if not api_key:
            raise RuntimeError(
                "A Gemini API key is required (set GOOGLE_API_KEY)."
            )

        self._genai_types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._timeout_s = timeout_s
        self._retries = retries
        self._retry_delay_s = retry_delay_s
        self._semaphore = semaphore

    @staticmethod
    def _mime_type(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        return mime_type or "image/jpeg"

    def _build_parts(self, image_paths: Sequence[Path]):
        parts = []
        for path in image_paths:
            with path.open("rb") as handle:
                parts.append(
                    self._genai_types.Part.from_bytes(
                        data=handle.read(),
                        mime_type=self._mime_type(path),
                    )
                )
        return parts

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        parts = [user_message, *self._build_parts(image_paths)]
        config = self._genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            top_p=self._top_p,
        )

        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with self._semaphore:
                    started = time.perf_counter()
                    response = await asyncio.wait_for(
                        self._client.aio.models.generate_content(
                            model=self._model,
                            contents=parts,
                            config=config,
                        ),
                        timeout=self._timeout_s,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                return ProviderResponse(
                    raw_output=str(response.text),
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:  # provider-specific transport failures
                last_error = exc
                if attempt >= self._retries:
                    break
                await asyncio.sleep(self._retry_delay_s * (attempt + 1))

        raise RuntimeError(f"Gemini request failed: {last_error}")

class ProviderFactory:
    """Create the selected model-provider strategy in one place."""

    @staticmethod
    def create(
        provider: str,
        *,
        host: str,
        model: str,
        temperature: float,
        top_p: float,
        num_ctx: int,
        timeout_s: float,
        retries: int,
        retry_delay_s: float,
        semaphore: asyncio.Semaphore,
        api_key: str | None = None,
    ) -> ModelProvider:
        name = provider.strip().lower()
        if name == "ollama":
            return OllamaVisionAdapter(
                host=host,
                model=model,
                temperature=temperature,
                top_p=top_p,
                num_ctx=num_ctx,
                timeout_s=timeout_s,
                retries=retries,
                retry_delay_s=retry_delay_s,
                semaphore=semaphore,
            )
        if name == "openai":
            return OpenAIVisionAdapter(
                host=host,
                api_key=api_key,
                model=model,
                temperature=temperature,
                top_p=top_p,
                num_ctx=num_ctx,
                timeout_s=timeout_s,
                retries=retries,
                retry_delay_s=retry_delay_s,
                semaphore=semaphore,
                    )
        if name == "gemini":
            return GeminiVisionAdapter(
                host=host,
                api_key=api_key,
                model=model,
                temperature=temperature,
                top_p=top_p,
                num_ctx=num_ctx,
                timeout_s=timeout_s,
                retries=retries,
                retry_delay_s=retry_delay_s,
                semaphore=semaphore,
                    )
        raise ValueError(
            f"Unsupported model provider: {provider!r}. "
            "Add a new adapter and register it in ProviderFactory."
        )
