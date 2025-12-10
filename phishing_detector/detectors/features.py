"""Feature extraction for phishing detection."""
from __future__ import annotations

import email
import email.policy
import re
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from google import genai

from phishing_detector.config import AppConfig


@dataclass
class FeatureExtractionResult:
    raw_text: str
    text_features: Dict[str, float]
    keyword_hits: List[str] = field(default_factory=list)
    llm_features: Dict[str, float] = field(default_factory=dict)


BASIC_KEYWORDS = [
    "密码重置",
    "账号冻结",
    "点击链接",
    "verify your account",
    "reset password",
]


URL_PATTERN = re.compile(r"https?://[\w./-]+", re.IGNORECASE)


def _html_to_text(html: str) -> str:
    """Convert basic HTML content to plain text."""
    # Remove script and style content first
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style.*?>.*?</style>", "", html, flags=re.IGNORECASE | re.DOTALL)
    # Strip remaining tags and collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_body_from_message(msg: email.message.EmailMessage) -> str:
    """Get a plain-text body from an EmailMessage, preferring text/plain."""
    plain_parts: List[str] = []
    html_parts: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            content_type = part.get_content_type()
            try:
                content = part.get_content()
            except Exception:
                content = None
            if not content:
                continue
            if content_type == "text/plain":
                plain_parts.append(str(content))
            elif content_type == "text/html":
                html_parts.append(str(content))
    else:
        try:
            content = msg.get_content()
        except Exception:
            content = None
        if content:
            if msg.get_content_type() == "text/plain":
                plain_parts.append(str(content))
            elif msg.get_content_type() == "text/html":
                html_parts.append(str(content))

    if plain_parts:
        return "\n".join(part.strip() for part in plain_parts if part)
    if html_parts:
        return _html_to_text("\n".join(html_parts))
    return ""


def _parse_eml(path: Path) -> Dict[str, str]:
    with path.open("rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    body = _extract_body_from_message(msg)

    return {
        "subject": msg.get("subject", ""),
        "from": msg.get("from", ""),
        "to": msg.get("to", ""),
        "body": body,
    }


def load_email(path: str | Path) -> Dict[str, str]:
    """Load email content from a .eml or .txt file into a structured dict."""
    email_path = Path(path)

    if not email_path.exists():
        raise FileNotFoundError(f"Email file not found: {email_path}")

    suffix = email_path.suffix.lower()
    if suffix == ".eml":
        return _parse_eml(email_path)
    if suffix == ".txt":
        text = email_path.read_text(encoding="utf-8", errors="ignore")
        return {"subject": "", "from": "", "to": "", "body": text}

    raise ValueError(f"Unsupported file type: {email_path.suffix}. Use .eml or .txt.")


def compute_text_features(text: str) -> Dict[str, float]:
    sentences = [s for s in re.split(r"[.!?]\s+", text) if s]
    urls = URL_PATTERN.findall(text)

    return {
        "length": float(len(text)),
        "avg_sentence_length": float(sum(len(s) for s in sentences) / len(sentences)) if sentences else 0.0,
        "exclamation_count": float(text.count("!")),
        "has_url": 1.0 if urls else 0.0,
        # TODO: implement real domain risk scoring and URL heuristics.
        "suspicious_domain_score": 0.0,
    }


def detect_keywords(text: str) -> List[str]:
    lowered = text.lower()
    return [kw for kw in BASIC_KEYWORDS if kw.lower() in lowered]


def extract_features_from_file(path: Path, cfg: AppConfig) -> FeatureExtractionResult:
    email_content = load_email(path)
    text = email_content.get("body", "")
    basic_features = extract_basic_features(email_content)
    text_features = {k: v for k, v in basic_features.items() if k != "phishing_keyword_hits"}
    keyword_hits = basic_features.get("phishing_keyword_hits", [])

    return FeatureExtractionResult(
        raw_text=text,
        text_features=text_features,
        keyword_hits=keyword_hits,
        llm_features={},
    )


def extract_basic_features(email: Dict[str, str]) -> Dict[str, object]:
    """Compute heuristic features from a structured email dict."""

    subject = email.get("subject", "")
    body = email.get("body", "")
    combined_text = f"{subject}\n{body}".strip()

    sentences = [s.strip() for s in re.split(r"[.!?]+\s*", combined_text) if s.strip()]
    urls = URL_PATTERN.findall(combined_text)
    keyword_hits = detect_keywords(combined_text)

    text_length = float(len(combined_text))
    sentence_count = float(len(sentences))
    avg_sentence_length = float(sum(len(s) for s in sentences) / len(sentences)) if sentences else 0.0

    return {
        "length": text_length,  # kept for backward compatibility
        "text_length": text_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "exclamation_count": float(combined_text.count("!")),
        "url_count": float(len(urls)),
        "has_url": 1.0 if urls else 0.0,
        "suspicious_domain_score": 0.0,  # placeholder for richer heuristics
        "phishing_keyword_count": float(len(keyword_hits)),
        "has_phishing_keywords": 1.0 if keyword_hits else 0.0,
        "phishing_keyword_hits": keyword_hits,
    }


def build_gemini_client(cfg: AppConfig | None = None) -> genai.Client:
    """Initialize the Gemini client from config or environment variables."""

    api_key = cfg.llm.api_key if cfg and cfg.llm.api_key else None
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 或 GOOGLE_API_KEY 未设置，无法调用 Gemini。")
    return genai.Client(api_key=api_key)


def extract_llm_features(email: Dict[str, str], cfg: AppConfig | None = None) -> Dict[str, object]:
    """Call Gemini to score LLM-likeness and phishing risk."""

    client = build_gemini_client(cfg)
    model = cfg.llm.model if cfg else "gemini-1.5-flash"
    temperature = cfg.llm.temperature if cfg else 0.0
    max_tokens = cfg.llm.max_tokens if cfg else 256

    subject = email.get("subject", "(no subject)")
    body = email.get("body", "")

    prompt = (
        "阅读下面的邮件（主题 + 正文），请返回 JSON，字段包括：\n"
        "- llm_like_score: 0到1之间，越高越像 LLM 生成的商务邮件\n"
        "- phishing_risk_score: 0到1之间，越高越可能是钓鱼\n"
        "- analysis: 简要中文说明\n"
        "只输出 JSON，不要额外解释。"
    )

    prompt_text = (
        "你是一名安全分析助手，返回结构化 JSON 评分。\n\n"
        f"{prompt}\n\nSubject: {subject}\n\nBody:\n{body}"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )

    content = response.text
    if not content:
        raise RuntimeError("LLM(Gemini) response was empty when extracting features.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to parse LLM JSON response: {exc}") from exc

    def _safe_float(key: str) -> float:
        value = data.get(key, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return {
        "llm_like_score": _safe_float("llm_like_score"),
        "phishing_risk_score": _safe_float("phishing_risk_score"),
        "llm_analysis": data.get("analysis", ""),
    }


def extract_all_features(email: Dict[str, str], use_llm: bool = True, cfg: AppConfig | None = None) -> Dict[str, object]:
    """Combine basic and (optionally) LLM-derived features into one flat dict."""

    features = extract_basic_features(email)
    if use_llm:
        try:
            llm_feats = extract_llm_features(email, cfg)
            features.update(llm_feats)
        except Exception as exc:
            features["llm_error"] = str(exc)
    return features
