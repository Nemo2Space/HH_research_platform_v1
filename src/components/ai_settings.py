"""
AI Model Settings Component

Provides:
1. Sidebar dropdown for global AI model selection
2. Multi-model comparison feature for analyzing stocks
3. Model status indicators

Usage in app.py:
    from src.components.ai_settings import render_ai_sidebar, render_ai_comparison

    # In sidebar
    with st.sidebar:
        render_ai_sidebar()
"""

import streamlit as st
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

# Import the models config
try:
    from src.components.ai_models_config import (
        get_all_models,
        get_model_api_id,
        get_model_display_name,
        get_grouped_models,
        OPENAI_MODELS,
        ANTHROPIC_MODELS,
        LOCAL_MODELS,
    )
    MODELS_CONFIG_AVAILABLE = True
except ImportError:
    MODELS_CONFIG_AVAILABLE = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# FALLBACK IF CONFIG NOT AVAILABLE
# =============================================================================

if not MODELS_CONFIG_AVAILABLE:
    # Minimal fallback models
    def get_all_models():
        return {
            "qwen_local": {"name": "Qwen 32B (Local)", "api_id": "local", "icon": "🏠", "provider": "qwen"},
            "gpt52": {"name": "GPT-5.2", "api_id": "gpt-5.2", "icon": "🧠", "provider": "openai", "api_key_env": "OPENAI_API_KEY"},
            "gpt4o": {"name": "GPT-4o", "api_id": "gpt-4o", "icon": "🧠", "provider": "openai", "api_key_env": "OPENAI_API_KEY"},
            "claude_sonnet": {"name": "Claude Sonnet 4", "api_id": "claude-sonnet-4-20250514", "icon": "🎭", "provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
        }

    def get_model_api_id(model_id):
        models = get_all_models()
        return models.get(model_id, {}).get("api_id", model_id)

    def get_model_display_name(model_id):
        models = get_all_models()
        if model_id in models:
            m = models[model_id]
            return f"{m.get('icon', '🤖')} {m.get('name', model_id)}"
        return model_id

    def get_grouped_models():
        return {"All Models": list(get_all_models().values())}


# =============================================================================
# MODEL AVAILABILITY CHECKING
# =============================================================================

def check_model_availability(model_id: str) -> Tuple[bool, str]:
    """Check if a model is available (API key set, server running, etc.)."""
    all_models = get_all_models()

    if model_id not in all_models:
        return False, "Unknown model"

    model = all_models[model_id]
    provider = model.get("provider", "unknown")
    api_key_env = model.get("api_key_env")

    # Check API key requirement
    if api_key_env:
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            return False, f"No API key ({api_key_env})"

    # Check provider-specific availability
    if provider == "qwen":
        base_url = os.getenv("LLM_QWEN_BASE_URL", "http://localhost:8090/v1")
        try:
            import requests
            resp = requests.get(f"{base_url.replace('/v1', '')}/health", timeout=2)
            if resp.status_code != 200:
                return False, "Server not responding"
        except:
            return False, "Server not reachable"

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            import requests
            resp = requests.get(f"{base_url}/api/tags", timeout=2)
            if resp.status_code != 200:
                return False, "Ollama not responding"
        except:
            return False, "Ollama not reachable"

    return True, "Available"


def get_available_models() -> List[str]:
    """Get list of available model IDs."""
    available = []
    for model_id in get_all_models():
        is_available, _ = check_model_availability(model_id)
        if is_available:
            available.append(model_id)
    return available


# =============================================================================
# SESSION STATE
# =============================================================================

def init_ai_session_state():
    """Initialize AI-related session state."""
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = "qwen_local"
    if 'ai_comparison_results' not in st.session_state:
        st.session_state.ai_comparison_results = {}


# =============================================================================
# SIDEBAR COMPONENT
# =============================================================================

def render_ai_sidebar():
    """Render AI model selection in sidebar."""
    init_ai_session_state()

    st.markdown("### 🤖 AI Model")

    all_models = get_all_models()
    current_model_id = st.session_state.ai_model

    # Build options list with availability status
    available_options = []
    option_labels = {}
    unavailable_info = []

    for model_id, model in all_models.items():
        is_available, status = check_model_availability(model_id)
        icon = model.get('icon', '🤖')
        name = model.get('name', model_id)

        if is_available:
            available_options.append(model_id)
            option_labels[model_id] = f"{icon} {name}"
        else:
            unavailable_info.append(f"{icon} {name}: {status}")

    if not available_options:
        st.error("No AI models available!")
        st.caption("Check your .env file and API keys.")
        return

    # Ensure current selection is valid
    if current_model_id not in available_options:
        current_model_id = available_options[0]
        st.session_state.ai_model = current_model_id

    # Model dropdown
    selected = st.selectbox(
        "Select AI Model",
        options=available_options,
        index=available_options.index(current_model_id),
        format_func=lambda x: option_labels.get(x, x),
        key="ai_model_selector",
        label_visibility="collapsed"
    )

    # Update if changed
    if selected != st.session_state.ai_model:
        st.session_state.ai_model = selected
        st.toast(f"Switched to {option_labels.get(selected, selected)}", icon="🔄")

    # Show model info
    if selected in all_models:
        model = all_models[selected]

        # Quality/Speed/Cost indicators
        quality_map = {"low": "⭐", "medium": "⭐⭐", "high": "⭐⭐⭐", "very_high": "⭐⭐⭐⭐"}
        speed_map = {"slow": "🐢", "medium": "🚶", "fast": "🚀", "very_fast": "⚡"}
        cost_map = {"free": "🆓", "low": "💵", "medium": "💵💵", "high": "💵💵💵"}

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Quality: {quality_map.get(model.get('quality', 'medium'), '⭐⭐')}")
        with col2:
            st.caption(f"Speed: {speed_map.get(model.get('speed', 'medium'), '🚶')}")
        with col3:
            st.caption(f"Cost: {cost_map.get(model.get('cost', 'medium'), '💵')}")

        st.caption(model.get('description', ''))

    # Show unavailable models in expander
    if unavailable_info:
        with st.expander(f"⚠️ {len(unavailable_info)} models unavailable", expanded=False):
            for info in unavailable_info:
                st.caption(info)


# =============================================================================
# AI RESPONSE FUNCTIONS
# =============================================================================

def get_ai_response_for_model(
    model_id: str,
    prompt: str,
    system_prompt: str = None,
) -> Tuple[str, str]:
    """
    Get AI response using a specific model.

    Returns:
        Tuple of (response_text, model_display_name)
    """
    all_models = get_all_models()

    if model_id not in all_models:
        return f"Unknown model: {model_id}", f"❌ {model_id}"

    model = all_models[model_id]
    provider = model.get("provider", "unknown")
    api_id = model.get("api_id", model_id)
    display_name = f"{model.get('icon', '🤖')} {model.get('name', model_id)}"

    # Check availability
    is_available, status = check_model_availability(model_id)
    if not is_available:
        return f"Model not available: {status}", f"❌ {display_name} ({status})"

    try:
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call appropriate provider
        if provider == "openai":
            return _call_openai(api_id, messages, display_name)
        elif provider == "anthropic":
            return _call_anthropic(api_id, messages, system_prompt, display_name)
        elif provider == "qwen":
            return _call_qwen(api_id, messages, display_name)
        elif provider == "ollama":
            return _call_ollama(api_id, messages, display_name)
        else:
            return f"Unknown provider: {provider}", f"❌ {display_name}"

    except Exception as e:
        logger.error(f"AI call failed for {model_id}: {e}")
        return f"Error: {str(e)[:100]}", f"❌ {display_name} (Error)"


def _call_openai(model: str, messages: list, display_name: str) -> Tuple[str, str]:
    """Call OpenAI API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, timeout=120)

    logger.info(f"Calling OpenAI: {model}")

    # GPT-5.x and o-series use max_completion_tokens, older models use max_tokens
    is_new_model = model.startswith(('gpt-5', 'o1', 'o3', 'o4'))

    if is_new_model:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.15,
            max_completion_tokens=3000
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.15,
            max_tokens=3000
        )

    result = response.choices[0].message.content

    # Clean thinking tags
    if '</think>' in result:
        result = result.split('</think>')[-1].strip()

    return result, display_name


def _call_anthropic(model: str, messages: list, system_prompt: str, display_name: str) -> Tuple[str, str]:
    """Call Anthropic API."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    logger.info(f"Calling Anthropic: {model}")

    # Convert messages for Claude
    claude_messages = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        system=system_prompt or "",
        messages=claude_messages
    )

    return response.content[0].text, display_name


def _call_qwen(model: str, messages: list, display_name: str) -> Tuple[str, str]:
    """Call local Qwen via llama.cpp."""
    from openai import OpenAI

    base_url = os.getenv("LLM_QWEN_BASE_URL", "http://localhost:8090/v1")
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=120)

    logger.info(f"Calling Qwen: {model}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.15,
        max_tokens=3000
    )

    result = response.choices[0].message.content

    # Clean thinking tags
    if '</think>' in result:
        result = result.split('</think>')[-1].strip()

    return result, display_name


def _call_ollama(model: str, messages: list, display_name: str) -> Tuple[str, str]:
    """Call Ollama API."""
    from openai import OpenAI

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama", timeout=120)

    logger.info(f"Calling Ollama: {model}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.15
    )

    result = response.choices[0].message.content

    # Clean thinking tags
    if '</think>' in result:
        result = result.split('</think>')[-1].strip()

    return result, display_name


# =============================================================================
# HELPER FUNCTIONS FOR EXTERNAL USE
# =============================================================================

def get_current_model_id() -> str:
    """Get the currently selected model ID."""
    init_ai_session_state()
    return st.session_state.ai_model


def get_current_model() -> dict:
    """Get the currently selected model config."""
    model_id = get_current_model_id()
    all_models = get_all_models()
    return all_models.get(model_id, {})


def get_current_model_response(prompt: str, system_prompt: str = None) -> Tuple[str, str]:
    """Get response from the currently selected model."""
    return get_ai_response_for_model(
        get_current_model_id(),
        prompt,
        system_prompt
    )