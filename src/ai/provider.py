"""
AI Provider Helper - Supports multiple AI backends.

This module provides a unified interface for different AI providers:
- Local Qwen (llama.cpp)
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Ollama (local models)
- Custom OpenAI-compatible endpoints

Usage:
    from src.ai.provider import get_ai_client, get_ai_response

    # Get streaming response
    for chunk in get_ai_response("What is the market outlook?"):
        print(chunk, end="")

    # Or use client directly
    client = get_ai_client()
    response = client.chat(messages=[...])
"""

import os
from typing import Generator, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import config
try:
    from src.config import config
except ImportError:
    # Fallback if config not available yet
    config = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """Chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> str:
        """Send chat messages and get response."""
        pass

    @abstractmethod
    def chat_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        """Send chat messages and stream response."""
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        """Check if provider is available."""
        pass


class QwenProvider(AIProvider):
    """Local Qwen via llama.cpp server."""

    def __init__(self):
        from openai import OpenAI

        self.base_url = config.ai.qwen_base_url if config else os.getenv("LLM_QWEN_BASE_URL", "http://localhost:8090/v1")
        self.model = config.ai.qwen_model if config else os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")

        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="not-needed",
                timeout=config.ai.timeout_seconds if config else 120
            )
            self._available = True
            logger.info(f"Qwen provider connected to {self.base_url}")
        except Exception as e:
            logger.error(f"Qwen provider init failed: {e}")
            self._available = False
            self.client = None

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages: List[Message], **kwargs) -> str:
        if not self.available:
            return "AI provider not available"

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get('temperature', config.ai.temperature if config else 0.15),
            max_tokens=kwargs.get('max_tokens', config.ai.max_tokens if config else 3000),
            top_p=kwargs.get('top_p', config.ai.top_p if config else 0.85),
        )

        return response.choices[0].message.content

    def chat_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.available:
            yield "AI provider not available"
            return

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get('temperature', config.ai.temperature if config else 0.15),
            max_tokens=kwargs.get('max_tokens', config.ai.max_tokens if config else 3000),
            top_p=kwargs.get('top_p', config.ai.top_p if config else 0.85),
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAIProvider(AIProvider):
    """OpenAI API provider (GPT-4, etc.)."""

    def __init__(self):
        from openai import OpenAI

        self.api_key = config.ai.openai_api_key if config else os.getenv("OPENAI_API_KEY", "")
        self.model = config.ai.openai_model if config else os.getenv("OPENAI_MODEL", "gpt-4o")
        self.base_url = config.ai.openai_base_url if config else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.api_key:
            logger.warning("OpenAI API key not set")
            self._available = False
            self.client = None
            return

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=config.ai.timeout_seconds if config else 120
            )
            self._available = True
            logger.info(f"OpenAI provider ready with model {self.model}")
        except Exception as e:
            logger.error(f"OpenAI provider init failed: {e}")
            self._available = False
            self.client = None

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages: List[Message], **kwargs) -> str:
        if not self.available:
            return "OpenAI provider not available - check API key"

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get('temperature', config.ai.temperature if config else 0.15),
            max_tokens=kwargs.get('max_tokens', config.ai.max_tokens if config else 3000),
        )

        return response.choices[0].message.content

    def chat_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.available:
            yield "OpenAI provider not available - check API key"
            return

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get('temperature', config.ai.temperature if config else 0.15),
            max_tokens=kwargs.get('max_tokens', config.ai.max_tokens if config else 3000),
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(AIProvider):
    """Anthropic API provider (Claude)."""

    def __init__(self):
        self.api_key = config.ai.anthropic_api_key if config else os.getenv("ANTHROPIC_API_KEY", "")
        self.model = config.ai.anthropic_model if config else os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

        if not self.api_key:
            logger.warning("Anthropic API key not set")
            self._available = False
            self.client = None
            return

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self._available = True
            logger.info(f"Anthropic provider ready with model {self.model}")
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            self._available = False
            self.client = None
        except Exception as e:
            logger.error(f"Anthropic provider init failed: {e}")
            self._available = False
            self.client = None

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages: List[Message], **kwargs) -> str:
        if not self.available:
            return "Anthropic provider not available - check API key"

        # Convert messages - Claude uses different format
        system_msg = ""
        claude_messages = []

        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                claude_messages.append({"role": m.role, "content": m.content})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', config.ai.max_tokens if config else 3000),
            system=system_msg,
            messages=claude_messages
        )

        return response.content[0].text

    def chat_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.available:
            yield "Anthropic provider not available - check API key"
            return

        # Convert messages
        system_msg = ""
        claude_messages = []

        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                claude_messages.append({"role": m.role, "content": m.content})

        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', config.ai.max_tokens if config else 3000),
            system=system_msg,
            messages=claude_messages
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaProvider(AIProvider):
    """Ollama local model provider."""

    def __init__(self):
        from openai import OpenAI

        self.base_url = config.ai.ollama_base_url if config else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = config.ai.ollama_model if config else os.getenv("OLLAMA_MODEL", "llama3.1:70b")

        try:
            # Ollama provides OpenAI-compatible API
            self.client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="ollama",
                timeout=config.ai.timeout_seconds if config else 120
            )
            self._available = True
            logger.info(f"Ollama provider connected to {self.base_url}")
        except Exception as e:
            logger.error(f"Ollama provider init failed: {e}")
            self._available = False
            self.client = None

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages: List[Message], **kwargs) -> str:
        if not self.available:
            return "Ollama provider not available"

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get('temperature', config.ai.temperature if config else 0.15),
        )

        return response.choices[0].message.content

    def chat_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.available:
            yield "Ollama provider not available"
            return

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get('temperature', config.ai.temperature if config else 0.15),
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Provider registry
_providers: Dict[str, type] = {
    "qwen": QwenProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}

# Cached provider instance
_current_provider: Optional[AIProvider] = None


def get_ai_provider() -> AIProvider:
    """Get the configured AI provider."""
    global _current_provider

    if _current_provider is not None:
        return _current_provider

    provider_name = config.ai.provider if config else os.getenv("AI_PROVIDER", "qwen")

    if provider_name not in _providers:
        logger.warning(f"Unknown provider '{provider_name}', falling back to qwen")
        provider_name = "qwen"

    provider_class = _providers[provider_name]
    _current_provider = provider_class()

    return _current_provider


def get_ai_client() -> AIProvider:
    """Alias for get_ai_provider()."""
    return get_ai_provider()


def get_ai_response(prompt: str, system_prompt: str = None, **kwargs) -> str:
    """Get AI response for a simple prompt."""
    provider = get_ai_provider()

    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=prompt))

    return provider.chat(messages, **kwargs)


def get_ai_response_stream(prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
    """Stream AI response for a simple prompt."""
    provider = get_ai_provider()

    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=prompt))

    yield from provider.chat_stream(messages, **kwargs)


def switch_provider(provider_name: str) -> AIProvider:
    """Switch to a different AI provider."""
    global _current_provider

    if provider_name not in _providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(_providers.keys())}")

    provider_class = _providers[provider_name]
    _current_provider = provider_class()

    logger.info(f"Switched to AI provider: {provider_name}")

    return _current_provider


def list_providers() -> List[str]:
    """List available AI providers."""
    return list(_providers.keys())


def get_provider_status() -> Dict[str, bool]:
    """Get availability status of all providers."""
    status = {}
    for name, provider_class in _providers.items():
        try:
            provider = provider_class()
            status[name] = provider.available
        except Exception:
            status[name] = False
    return status


if __name__ == "__main__":
    # Test providers
    print("AI Provider Status:")
    print("="*40)
    for name, available in get_provider_status().items():
        status = "✅" if available else "❌"
        print(f"  {status} {name}")

    print("\nCurrent provider:", config.ai.provider if config else os.getenv("AI_PROVIDER", "qwen"))

    # Test current provider
    provider = get_ai_provider()
    if provider.available:
        print("\nTest response:")
        response = get_ai_response("Say 'Hello from AI!' in exactly those words.")
        print(response)