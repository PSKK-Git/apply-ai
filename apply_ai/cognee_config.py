"""Configure Cognee (the knowledge spine) from environment.

Cognee reads most of its own settings from standard env vars (LLM_API_KEY,
LLM_PROVIDER, LLM_MODEL, EMBEDDING_PROVIDER, EMBEDDING_API_KEY, ...). This module
loads the project's .env, applies a few sane defaults, honours optional
APPLY_AI_* overrides, and can route to Cognee Cloud. It runs exactly once.
"""
from __future__ import annotations
import os
from functools import lru_cache
from dotenv import load_dotenv


def _apply(setter_name: str, env_name: str) -> None:
    val = os.getenv(env_name)
    if not val:
        return
    import cognee
    getattr(cognee.config, setter_name)(val)


@lru_cache(maxsize=1)
def configure() -> None:
    """Idempotent: load .env, set defaults, apply overrides, optional cloud."""
    load_dotenv()
    # Single-user embedded mode unless the operator opts into multi-tenant.
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    # Optional explicit overrides (otherwise cognee reads its own env vars).
    _apply("set_llm_provider", "APPLY_AI_LLM_PROVIDER")
    _apply("set_llm_api_key", "APPLY_AI_LLM_API_KEY")
    _apply("set_llm_model", "APPLY_AI_LLM_MODEL")
    _apply("set_embedding_provider", "APPLY_AI_EMBEDDING_PROVIDER")
    _apply("set_embedding_api_key", "APPLY_AI_EMBEDDING_API_KEY")
    _apply("set_embedding_model", "APPLY_AI_EMBEDDING_MODEL")

    # Cognee Cloud (optional): offloads compute to the hosted service.
    if os.getenv("APPLY_AI_COGNEE_CLOUD", "").lower() in ("1", "true", "yes"):
        import cognee
        cognee.serve(api_key=os.environ["COGNEE_API_KEY"])
