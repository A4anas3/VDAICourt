import os
from typing import Optional, Literal, List, Tuple
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Load environment variables from .env file
load_dotenv()
os.environ.pop("GEMINI_API_KEY", None)


def get_all_google_api_keys() -> List[str]:
    """
    Extracts all configured Gemini / Google API keys from .env environment variables.
    Supports GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, etc.
    """
    keys = []
    # Check indexed keys GOOGLE_API_KEY_1 to GOOGLE_API_KEY_10
    for idx in range(1, 10):
        k = os.getenv(f"GOOGLE_API_KEY_{idx}") or os.getenv(f"GEMINI_API_KEY_{idx}")
        if k and k.strip():
            keys.append(k.strip())

    # Fallback to single default keys if indexed keys not present
    if not keys:
        default_k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if default_k and default_k.strip():
            keys.append(default_k.strip())

    # Remove duplicates while maintaining order
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


def init_model(
    provider: Optional[Literal["gemini", "ollama"]] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Initializes and returns a LangChain Chat Model (Gemini or Ollama).
    """
    if temperature is None:
        try:
            temperature = float(os.getenv("TEMPERATURE", "0.0"))
        except ValueError:
            temperature = 0.0

    if not provider:
        use_ollama_env = os.getenv("USE_OLLAMA", "false").strip().lower()
        provider = "ollama" if use_ollama_env in ("true", "1", "yes") else "gemini"

    if provider == "gemini":
        selected_model = model_name or os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
        target_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not target_api_key:
            keys = get_all_google_api_keys()
            if keys:
                target_api_key = keys[0]

        if not target_api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing in .env!"
            )

        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=target_api_key,
            temperature=temperature,
            request_timeout=float(os.getenv("API_TIMEOUT", "220.0")),
            **kwargs
        )

    elif provider == "ollama":
        selected_model = (
            model_name
            or os.getenv("OLLAMA_MODEL")
            or os.getenv("OLLAMA_OCR_MODEL", "qwen2.5-vl")
        )
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=selected_model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model provider: '{provider}'. Choose 'gemini' or 'ollama'.")


def init_model_pool(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> List[Tuple[str, BaseChatModel]]:
    """
    Initializes a pool of model instances, one for each configured API key.
    Returns list of (api_key_name, model_instance) tuples.
    """
    keys = get_all_google_api_keys()
    if not keys:
        # Fallback single model
        m = init_model(model_name=model_name, temperature=temperature, **kwargs)
        return [("API_KEY_1", m)]

    pool = []
    for idx, key in enumerate(keys, start=1):
        m = init_model(model_name=model_name, temperature=temperature, api_key=key, **kwargs)
        pool.append((f"API_KEY_{idx}", m))
    
    print(f"[ModelPool] Initialized pool with {len(pool)} API Key(s)")
    return pool


if __name__ == "__main__":
    print("=== Testing Multi-Key Model Pool Initialization ===")
    keys = get_all_google_api_keys()
    print(f"Configured API Keys count: {len(keys)}")
    pool = init_model_pool()
    for name, model in pool:
        print(f"  - {name}: {type(model).__name__}")
