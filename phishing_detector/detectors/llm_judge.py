"""LLM-assisted phishing judgement utilities powered by an OpenAI-compatible proxy."""
from __future__ import annotations

import json
from typing import Dict

from phishing_detector.config import AppConfig
from phishing_detector.detectors.llm_client import build_proxy_openai_client


def _prepare_messages(email: Dict[str, str]) -> list[Dict[str, str]]:
    subject = email.get("subject", "(no subject)")
    body = email.get("body", "")

    instructions = (
        "阅读下面的邮件（主题 + 正文），判断它是否为钓鱼邮件。请严格输出 JSON：\n"
        "{\n  \"label\": \"phishing 或 benign\",\n  \"score\": 0到1之间的数字，越高越可能是钓鱼,\n  \"reason\": \"简短中文说明\"\n}"
    )

    return [
        {"role": "system", "content": "你是一名安全分析助手，必须输出 JSON。"},
        {
            "role": "user",
            "content": f"{instructions}\n\nSubject: {subject}\n\nBody:\n{body}",
        },
    ]


def _safe_label(raw_label: str | None) -> str:
    if not raw_label:
        return "unknown"
    normalized = str(raw_label).strip().lower()
    if normalized in {"phishing", "benign"}:
        return normalized
    return "unknown"


def llm_judge(email: Dict[str, str], cfg: AppConfig) -> Dict[str, object]:
    """Call the OpenAI-compatible proxy to obtain a phishing judgement with structured JSON."""

    client = build_proxy_openai_client(cfg)
    model = cfg.llm.model
    temperature = cfg.llm.temperature
    max_tokens = cfg.llm.max_tokens

    prompt_text = "\n\n".join(
        [
            "你是一名安全分析助手，必须输出 JSON。",
            _prepare_messages(email)[1]["content"],
        ]
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一名安全分析助手，必须输出 JSON。"},
                {"role": "user", "content": prompt_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - network/API failure
        return {
            "label": "unknown",
            "score": 0.0,
            "reason": f"LLM call failed: {exc}",
            "triggered_features": [],
        }

    content = response.choices[0].message.content if response.choices else ""
    if not content:
        return {
            "label": "unknown",
            "score": 0.0,
            "reason": "LLM response was empty.",
            "triggered_features": [],
        }

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        return {
            "label": "unknown",
            "score": 0.0,
            "reason": f"Failed to parse LLM JSON response: {exc}",
            "triggered_features": [],
        }

    def _safe_score(key: str) -> float:
        value = data.get(key, 0.0)
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    return {
        "label": _safe_label(data.get("label")),
        "score": _safe_score("score"),
        "reason": data.get("reason", ""),
        "triggered_features": [],
    }
