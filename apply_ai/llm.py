"""Provider-agnostic text generation for JD extraction + tailoring.

The generator is chosen by APPLY_AI_GEN_PROVIDER (default: gemini, the only key
with working quota in this environment). Flip to openai/anthropic via that env var
once those accounts have credit — no code change needed.
"""
from __future__ import annotations
import os


def complete(prompt: str, *, max_tokens: int = 1200) -> str:
    provider = os.getenv("APPLY_AI_GEN_PROVIDER", "gemini").lower()

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(os.getenv("APPLY_AI_GEN_MODEL", "gemini-2.5-flash"))
        return model.generate_content(prompt).text

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or os.environ["LLM_API_KEY"])
        resp = client.chat.completions.create(
            model=os.getenv("APPLY_AI_GEN_MODEL", "gpt-4o-mini"),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    if provider in ("anthropic", "claude"):
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=os.getenv("APPLY_AI_GEN_MODEL", "claude-sonnet-5"),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    if provider == "mistral":
        # Mistral's API is OpenAI-compatible.
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["MISTRAL_API_KEY"],
                        base_url="https://api.mistral.ai/v1")
        resp = client.chat.completions.create(
            model=os.getenv("APPLY_AI_GEN_MODEL", "mistral-small-latest"),
            max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

    if provider == "ollama":
        # Free / local. Ollama serves an OpenAI-compatible endpoint (default :11434/v1);
        # OLLAMA_BASE_URL can point at a remote box. No real key needed.
        from openai import OpenAI
        client = OpenAI(api_key="ollama",
                        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        resp = client.chat.completions.create(
            model=os.getenv("APPLY_AI_GEN_MODEL", "llama3.1"),
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

    if provider in ("custom", "openrouter", "litellm"):
        # Any OpenAI-compatible gateway (OpenRouter, a LiteLLM proxy, …).
        # Needs a base URL + key + a TEXT model (image models won't work here).
        import time
        from openai import OpenAI
        base = os.getenv("APPLY_AI_CUSTOM_BASE_URL") or "https://openrouter.ai/api/v1"
        client = OpenAI(api_key=os.getenv("APPLY_AI_CUSTOM_KEY", ""), base_url=base)
        primary = os.getenv("APPLY_AI_GEN_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        # free gateway models rate-limit constantly; try a few, one retry each
        models = list(dict.fromkeys([
            primary,
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
        ]))
        last = None
        for m in models:
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=m, max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}])
                    return resp.choices[0].message.content
                except Exception as e:
                    last = e
                    if "429" in str(e) and attempt == 0:
                        time.sleep(1.5)
                        continue
                    break
        raise last

    raise ValueError(f"unknown APPLY_AI_GEN_PROVIDER={provider!r}")
