"""
AI Models Configuration

This file defines all available AI models. To add a new model:
1. Add it to the appropriate provider section below
2. That's it! The model will automatically appear in the UI

Models are organized by provider for easy management.
"""

# =============================================================================
# OPENAI MODELS
# =============================================================================
OPENAI_MODELS = {
    # GPT-5.x Series (Latest)
    "gpt52_pro": {
        "name": "GPT-5.2 Pro",
        "api_id": "gpt-5.2-pro",
        "description": "Highest quality, best for complex analysis",
        "icon": "👑",
        "speed": "slow",
        "quality": "very_high",
        "cost": "high",
    },
    "gpt52": {
        "name": "GPT-5.2",
        "api_id": "gpt-5.2",
        "description": "Latest flagship - Professional work, long context",
        "icon": "🧠",
        "speed": "medium",
        "quality": "very_high",
        "cost": "medium",
    },
    "gpt51": {
        "name": "GPT-5.1",
        "api_id": "gpt-5.1",
        "description": "Fast responses, great steerability",
        "icon": "🧠",
        "speed": "fast",
        "quality": "very_high",
        "cost": "medium",
    },
    "gpt5": {
        "name": "GPT-5",
        "api_id": "gpt-5",
        "description": "Smarter, faster, more reliable",
        "icon": "🧠",
        "speed": "medium",
        "quality": "very_high",
        "cost": "medium",
    },
    "gpt5_mini": {
        "name": "GPT-5 Mini",
        "api_id": "gpt-5-mini",
        "description": "Smaller GPT-5, faster and cheaper",
        "icon": "⚡",
        "speed": "fast",
        "quality": "high",
        "cost": "low",
    },

    # GPT-4.x Series
    "gpt41": {
        "name": "GPT-4.1",
        "api_id": "gpt-4.1",
        "description": "Coding specialist, great instruction following",
        "icon": "💻",
        "speed": "fast",
        "quality": "high",
        "cost": "low",
    },
    "gpt41_mini": {
        "name": "GPT-4.1 Mini",
        "api_id": "gpt-4.1-mini",
        "description": "Fast, capable, efficient small model",
        "icon": "⚡",
        "speed": "fast",
        "quality": "medium",
        "cost": "low",
    },
    "gpt4o": {
        "name": "GPT-4o",
        "api_id": "gpt-4o",
        "description": "Previous flagship, multimodal",
        "icon": "🧠",
        "speed": "fast",
        "quality": "high",
        "cost": "low",
    },
    "gpt4o_mini": {
        "name": "GPT-4o Mini",
        "api_id": "gpt-4o-mini",
        "description": "Fast and cheap for simple tasks",
        "icon": "⚡",
        "speed": "fast",
        "quality": "medium",
        "cost": "low",
    },

    # o-Series (Reasoning)
    "o4_mini": {
        "name": "o4-mini",
        "api_id": "o4-mini",
        "description": "Fast reasoning, math & coding specialist",
        "icon": "🔮",
        "speed": "medium",
        "quality": "very_high",
        "cost": "medium",
    },
    "o3_mini": {
        "name": "o3-mini",
        "api_id": "o3-mini",
        "description": "Reasoning model, configurable effort",
        "icon": "🔮",
        "speed": "medium",
        "quality": "high",
        "cost": "medium",
    },
}

# =============================================================================
# ANTHROPIC MODELS
# =============================================================================
ANTHROPIC_MODELS = {
    "claude_opus": {
        "name": "Claude Opus 4.5",
        "api_id": "claude-opus-4-5-20251101",
        "description": "Most capable Claude, complex analysis",
        "icon": "👑",
        "speed": "slow",
        "quality": "very_high",
        "cost": "high",
    },
    "claude_sonnet": {
        "name": "Claude Sonnet 4",
        "api_id": "claude-sonnet-4-20250514",
        "description": "Balanced performance and speed",
        "icon": "🎭",
        "speed": "medium",
        "quality": "very_high",
        "cost": "medium",
    },
    "claude_haiku": {
        "name": "Claude Haiku 4.5",
        "api_id": "claude-haiku-4-5-20251001",
        "description": "Fast and efficient for quick tasks",
        "icon": "🚀",
        "speed": "fast",
        "quality": "high",
        "cost": "low",
    },
}

# =============================================================================
# LOCAL MODELS (No API key required)
# =============================================================================
LOCAL_MODELS = {
    "qwen_local": {
        "name": "Qwen 32B (Local)",
        "api_id": "Qwen3-32B-Q6_K.gguf",  # Will be overridden by env var
        "description": "Local Qwen via llama.cpp - Free, private",
        "icon": "🏠",
        "provider": "qwen",
        "speed": "fast",
        "quality": "high",
        "cost": "free",
    },
    "ollama_llama": {
        "name": "Llama 3.1 70B (Ollama)",
        "api_id": "llama3.1:70b",
        "description": "Local Llama via Ollama - Open source",
        "icon": "🦙",
        "provider": "ollama",
        "speed": "medium",
        "quality": "high",
        "cost": "free",
    },
    "ollama_qwen": {
        "name": "Qwen 2.5 72B (Ollama)",
        "api_id": "qwen2.5:72b",
        "description": "Local Qwen via Ollama",
        "icon": "🏠",
        "provider": "ollama",
        "speed": "medium",
        "quality": "high",
        "cost": "free",
    },
    "ollama_deepseek": {
        "name": "DeepSeek V3 (Ollama)",
        "api_id": "deepseek-v3:latest",
        "description": "DeepSeek reasoning model via Ollama",
        "icon": "🔬",
        "provider": "ollama",
        "speed": "medium",
        "quality": "high",
        "cost": "free",
    },
}

# =============================================================================
# OTHER PROVIDERS (Easy to add more)
# =============================================================================
OTHER_MODELS = {
    # Google Gemini (if you want to add)
    # "gemini_pro": {
    #     "name": "Gemini 2.0 Pro",
    #     "api_id": "gemini-2.0-pro",
    #     "description": "Google's flagship model",
    #     "icon": "💎",
    #     "provider": "google",
    #     "speed": "medium",
    #     "quality": "very_high",
    #     "cost": "medium",
    # },

    # Groq (super fast inference)
    # "groq_llama": {
    #     "name": "Llama 3.1 70B (Groq)",
    #     "api_id": "llama-3.1-70b-versatile",
    #     "description": "Ultra-fast inference via Groq",
    #     "icon": "⚡",
    #     "provider": "groq",
    #     "speed": "very_fast",
    #     "quality": "high",
    #     "cost": "low",
    # },

    # Together AI
    # "together_mixtral": {
    #     "name": "Mixtral 8x22B (Together)",
    #     "api_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    #     "description": "Mixtral via Together AI",
    #     "icon": "🌪️",
    #     "provider": "together",
    #     "speed": "fast",
    #     "quality": "high",
    #     "cost": "low",
    # },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_models() -> dict:
    """Get all models combined into a single dict."""
    all_models = {}

    # Add OpenAI models
    for model_id, config in OPENAI_MODELS.items():
        all_models[model_id] = {**config, "provider": "openai", "api_key_env": "OPENAI_API_KEY"}

    # Add Anthropic models
    for model_id, config in ANTHROPIC_MODELS.items():
        all_models[model_id] = {**config, "provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}

    # Add Local models
    for model_id, config in LOCAL_MODELS.items():
        provider = config.get("provider", "local")
        all_models[model_id] = {**config, "provider": provider, "api_key_env": None}

    # Add Other models
    for model_id, config in OTHER_MODELS.items():
        all_models[model_id] = config

    return all_models


def get_models_by_provider(provider: str) -> dict:
    """Get models for a specific provider."""
    all_models = get_all_models()
    return {k: v for k, v in all_models.items() if v.get("provider") == provider}


def get_model_api_id(model_id: str) -> str:
    """Get the API model string for a model ID."""
    all_models = get_all_models()
    if model_id in all_models:
        return all_models[model_id].get("api_id", model_id)
    return model_id


def get_model_display_name(model_id: str) -> str:
    """Get display name with icon for a model."""
    all_models = get_all_models()
    if model_id in all_models:
        model = all_models[model_id]
        return f"{model.get('icon', '🤖')} {model.get('name', model_id)}"
    return model_id


# =============================================================================
# MODEL GROUPS (for UI organization)
# =============================================================================

MODEL_GROUPS = {
    "🏠 Local (Free)": ["qwen_local", "ollama_llama", "ollama_qwen", "ollama_deepseek"],
    "🧠 OpenAI Latest": ["gpt52_pro", "gpt52", "gpt51", "gpt5", "gpt5_mini"],
    "💻 OpenAI Coding": ["gpt41", "gpt41_mini"],
    "🔮 OpenAI Reasoning": ["o4_mini", "o3_mini"],
    "📦 OpenAI Legacy": ["gpt4o", "gpt4o_mini"],
    "🎭 Anthropic": ["claude_opus", "claude_sonnet", "claude_haiku"],
}


def get_grouped_models() -> dict:
    """Get models organized by group for UI display."""
    all_models = get_all_models()
    grouped = {}

    for group_name, model_ids in MODEL_GROUPS.items():
        grouped[group_name] = []
        for model_id in model_ids:
            if model_id in all_models:
                grouped[group_name].append({
                    "id": model_id,
                    **all_models[model_id]
                })

    return grouped


# =============================================================================
# PRINT AVAILABLE MODELS (for debugging)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AVAILABLE AI MODELS")
    print("="*60)

    for group_name, models in get_grouped_models().items():
        print(f"\n{group_name}")
        print("-" * 40)
        for model in models:
            icon = model.get('icon', '🤖')
            name = model.get('name', model['id'])
            api_id = model.get('api_id', 'N/A')
            quality = model.get('quality', 'N/A')
            cost = model.get('cost', 'N/A')
            print(f"  {icon} {name}")
            print(f"      API: {api_id} | Quality: {quality} | Cost: {cost}")

    print("\n" + "="*60)
    print(f"Total models: {len(get_all_models())}")
    print("="*60)