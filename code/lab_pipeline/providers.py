"""Model-provider strategies and adapters.

Strategy: every provider implements ModelProvider.infer().
Adapter: OllamaVisionAdapter / OpenAIVisionAdapter convert our common
request into each provider's own format.
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
    """Simple container holding the result of one model call.

    Attributes:
        raw_output: The raw text/string reply returned by the model.
        elapsed_ms: How long the call took to run, in milliseconds.
    """

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
        """Send a system prompt, a user message, and images to a model.

        Every concrete provider (Ollama, OpenAI, Gemini, ...) must
        implement this method so the rest of the pipeline can call any
        provider the same way, without caring which one it is.

        Args:
            system_prompt: Instructions that set the model's behavior.
            user_message: The actual question/prompt for this request.
            image_paths: Paths to the image file(s) to send along.

        Returns:
            A ProviderResponse with the model's raw output and timing.
        """
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
        """Set up the Ollama async client and store call settings.

        Args:
            host: URL of the Ollama server to connect to.
            model: Name of the Ollama model to use (e.g. a llama3.2 tag).
            temperature: Sampling temperature passed to the model.
            top_p: Nucleus-sampling top_p value passed to the model.
            num_ctx: Context window size (in tokens) for the model.
            timeout_s: How long to wait for a response before timing out.
            retries: How many extra attempts to make if a call fails.
            retry_delay_s: Base delay (seconds) between retry attempts.
            semaphore: Limits how many requests run at the same time.

        Raises:
            RuntimeError: If the 'ollama' Python package is not installed.
        """
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
        """Read an image file from disk and return it as base64 text.

        Args:
            path: Path to the image file on disk.

        Returns:
            The image bytes encoded as a base64 UTF-8 string, which is
            the format Ollama expects for image input.
        """
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        """Send one chat request (with optional images) to Ollama.

        Builds the Ollama-style message list, calls the model, retries
        on failure with a growing delay, and times how long the final
        successful call took.

        Args:
            system_prompt: Instructions that set the model's behavior.
            user_message: The actual question/prompt for this request.
            image_paths: Paths to image file(s) to attach to the message.

        Returns:
            A ProviderResponse containing the model's text reply and
            how many milliseconds the successful call took.

        Raises:
            RuntimeError: If every attempt (including retries) fails.
        """
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
        reasoning_effort: str | None = None,
    ) -> None:
        """Set up the OpenAI async client and store call settings.

        Args:
            host: Base URL for the OpenAI-compatible API endpoint.
            api_key: OpenAI API key; required, read from OPENAI_API_KEY.
            model: Name of the OpenAI model to use (e.g. "gpt-4o-mini").
            temperature: Sampling temperature passed to the model.
            top_p: Nucleus-sampling top_p value passed to the model.
            num_ctx: Unused by OpenAI; kept only so all adapters share
                the same constructor shape.
            timeout_s: How long to wait for a response before timing out.
            retries: How many extra attempts to make if a call fails.
            retry_delay_s: Base delay (seconds) between retry attempts.
            semaphore: Limits how many requests run at the same time.
            reasoning_effort: Optional reasoning-effort setting for
                models that support it (e.g. "low"/"medium"/"high").

        Raises:
            RuntimeError: If the 'openai' package is missing, or if no
                api_key was provided.
        """
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
        self._reasoning_effort = reasoning_effort

    @staticmethod
    def _encode_image_data_uri(path: Path) -> str:
        """Read an image file and turn it into a base64 data: URI.

        Args:
            path: Path to the image file on disk.

        Returns:
            A string like "data:image/jpeg;base64,....", which is the
            format OpenAI's vision input expects.
        """
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
        """Send one chat completion request (with optional images) to OpenAI.

        Builds the OpenAI-style message list, calls the model, and
        retries on failure. Rate-limit errors (HTTP 429) get a much
        longer exponential backoff than other transient errors.

        Args:
            system_prompt: Instructions that set the model's behavior.
            user_message: The actual question/prompt for this request.
            image_paths: Paths to image file(s) to attach to the message.

        Returns:
            A ProviderResponse containing the model's text reply and
            how many milliseconds the successful call took.

        Raises:
            RuntimeError: If every attempt (including retries) fails.
        """
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
                    request_kwargs = {
                        "model": self._model,
                        "messages": messages,
                        "temperature": self._temperature,
                        "top_p": self._top_p,
                    }
                    if self._reasoning_effort is not None:
                        request_kwargs["reasoning_effort"] = self._reasoning_effort
                    response = await self._client.chat.completions.create(
                        **request_kwargs
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
                is_rate_limited = "429" in str(exc) or "rate_limit" in str(exc).lower()
                if is_rate_limited:
                    # Back off much longer for rate limits than for other
                    # transient errors, with exponential growth + jitter.
                    delay = self._retry_delay_s * (5 ** (attempt + 1))
                else:
                    delay = self._retry_delay_s * (attempt + 1)
                await asyncio.sleep(delay)

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
        thinking_level: str = "minimal",
    ) -> None:
        """Set up the Gemini client and store call settings.

        Args:
            host: Unused by the google-genai SDK; kept only so all
                adapters share the same constructor shape.
            api_key: Gemini API key; required, read from GOOGLE_API_KEY.
            model: Name of the Gemini model to use (e.g. "gemini-2.0-flash").
            temperature: Sampling temperature passed to the model.
            top_p: Nucleus-sampling top_p value passed to the model.
            num_ctx: Unused by Gemini; kept only for interface parity.
            timeout_s: How long to wait for a response before timing out.
            retries: How many extra attempts to make if a call fails.
            retry_delay_s: Base delay (seconds) between retry attempts.
            semaphore: Limits how many requests run at the same time.
            thinking_level: How much internal "thinking" the model may
                do before answering (e.g. "minimal", "low", "high").

        Raises:
            RuntimeError: If the 'google-genai' package is missing, or
                if no api_key was provided.
        """
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
        self._thinking_level = thinking_level

    @staticmethod
    def _mime_type(path: Path) -> str:
        """Guess the MIME type (e.g. "image/png") of a file from its name.

        Args:
            path: Path to the file whose type should be guessed.

        Returns:
            The guessed MIME type string, or "image/jpeg" as a fallback
            if it cannot be guessed.
        """
        mime_type, _ = mimetypes.guess_type(path.name)
        return mime_type or "image/jpeg"

    def _build_parts(self, image_paths: Sequence[Path]):
        """Read each image file and wrap it as a Gemini content "Part".

        Args:
            image_paths: Paths to the image file(s) to include.

        Returns:
            A list of google-genai Part objects, one per image, ready
            to be included alongside the text prompt in a request.
        """
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
        """Send one generate-content request (with optional images) to Gemini.

        Builds the prompt parts and generation config, calls the model
        with a timeout, and retries on failure. Rate-limit / resource-
        exhausted errors get a much longer exponential backoff than
        other transient errors.

        Args:
            system_prompt: Instructions that set the model's behavior.
            user_message: The actual question/prompt for this request.
            image_paths: Paths to image file(s) to attach to the request.

        Returns:
            A ProviderResponse containing the model's text reply and
            how many milliseconds the successful call took.

        Raises:
            RuntimeError: If every attempt (including retries) fails.
        """
        parts = [user_message, *self._build_parts(image_paths)]
        config = self._genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            top_p=self._top_p,
            thinking_config=self._genai_types.ThinkingConfig(
                thinking_level=self._thinking_level,
            ),
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
                is_rate_limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if is_rate_limited:
                    delay = self._retry_delay_s * (5 ** (attempt + 1))
                else:
                    delay = self._retry_delay_s * (attempt + 1)
                await asyncio.sleep(delay)

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
        thinking_level: str = "minimal",
        reasoning_effort: str | None = None,
    ) -> ModelProvider:
        """Build and return the right adapter for the requested provider.

        This is the single place that decides, based on the `provider`
        name (e.g. "ollama", "openai", "gemini"), which adapter class to
        instantiate and how to pass along all the shared settings.

        Args:
            provider: Name of the provider to create ("ollama", "openai",
                or "gemini"; matching is case-insensitive).
            host: Server URL / base URL for the provider's API.
            model: Name/ID of the model to use.
            temperature: Sampling temperature passed to the model.
            top_p: Nucleus-sampling top_p value passed to the model.
            num_ctx: Context window size (only meaningful for Ollama).
            timeout_s: How long to wait for a response before timing out.
            retries: How many extra attempts to make if a call fails.
            retry_delay_s: Base delay (seconds) between retry attempts.
            semaphore: Limits how many requests run at the same time.
            api_key: API key for providers that need one (OpenAI/Gemini).
            thinking_level: Gemini-specific "thinking" effort setting.
            reasoning_effort: OpenAI-specific reasoning effort setting.

        Returns:
            A ready-to-use ModelProvider instance (one of
            OllamaVisionAdapter, OpenAIVisionAdapter, GeminiVisionAdapter).

        Raises:
            ValueError: If `provider` does not match any supported
                provider name.
        """
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
                reasoning_effort=reasoning_effort,
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
                thinking_level=thinking_level,
            )
        raise ValueError(
            f"Unsupported model provider: {provider!r}. "
            "Add a new adapter and register it in ProviderFactory."
        )
