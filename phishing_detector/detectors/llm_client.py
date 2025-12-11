"""Utilities to configure OpenAI-compatible clients (proxy or direct)."""
from __future__ import annotations

import os
import openai

from phishing_detector.config import AppConfig


def build_proxy_openai_client(cfg: AppConfig | None = None):
    """
    Initialize the OpenAI-compatible proxy client.

    Uses PROXY_API_KEY or OPENAI_API_KEY for authentication, and cfg.llm.base_url or
    the default https://api.gpt.ge/v1/ for the API endpoint.
    """

    api_key = cfg.llm.api_key if cfg and cfg.llm.api_key else None
    api_key = api_key or os.getenv("PROXY_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量中设置 PROXY_API_KEY 或 OPENAI_API_KEY。")

    base_url = getattr(cfg.llm, "base_url", None) if cfg else None
    base_url = base_url or os.getenv("PROXY_BASE_URL") or "https://api.gpt.ge/v1/"

    openai.api_key = api_key
    openai.base_url = base_url
    openai.default_headers = {"x-foo": "true"}
    return openai


def build_direct_openai_client(cfg: AppConfig | None = None):
    """Configure the official OpenAI client with direct API access."""

    api_key = cfg.llm.api_key if cfg and cfg.llm.api_key else None
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("使用 openai 直连模式时，需要设置 OPENAI_API_KEY。")

    openai.api_key = api_key
    openai.base_url = "https://api.openai.com/v1/"
    return openai


def build_llm_client(cfg: AppConfig | None = None):
    """Dispatch to proxy or direct OpenAI client based on configuration."""

    cfg = cfg or AppConfig()
    mode = getattr(cfg, "llm_client", "trans")
    if mode == "openai":
        return build_direct_openai_client(cfg)
    return build_proxy_openai_client(cfg)
