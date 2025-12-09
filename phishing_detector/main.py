"""CLI entry point for phishing detection prototype.

Usage:
    python phishing_detector/main.py path/to/email.eml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Allow running as a script (e.g., `python phishing_detector/main.py ...`).
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from phishing_detector import config
from phishing_detector.detectors.features import (
    FeatureExtractionResult,
    extract_basic_features,
    load_email,
)
from phishing_detector.detectors.llm_judge import llm_judge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phishing email detection CLI")
    parser.add_argument("email_path", type=Path, help="Path to .eml or .txt email file")
    parser.add_argument(
        "--mode",
        choices=["llm", "clf", "hybrid"],
        default="llm",
        help="Detection mode: llm-only, clf-only, or hybrid blend",
    )
    return parser.parse_args()


def _has_llm_key(cfg: config.AppConfig) -> bool:
    """Check if an API key is available for LLM features/judgement."""

    return bool(cfg.llm.api_key or os.getenv("OPENAI_API_KEY"))


def render_output(
    console: Console,
    overall: Dict[str, Any],
    llm_result: Optional[Dict[str, Any]],
    clf_result: Optional[Dict[str, Any]],
    feature_result: FeatureExtractionResult,
) -> None:
    """Render detection results with rich formatting."""

    label = overall.get("label", "unknown")
    score = float(overall.get("score", 0.0))
    color = "red" if label == "phishing" else "green"
    header = Text(f"检测结果: {label} | 总体风险分数: {score:.2f}", style=f"bold {color}")

    table = Table(title="模式结果", header_style="bold cyan")
    table.add_column("模式", justify="center")
    table.add_column("Label", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Reason", justify="left")

    def _fmt_row(result: Optional[Dict[str, Any]]) -> tuple[str, str, str]:
        if not result:
            return ("N/A", "-", "")
        return (
            str(result.get("label", "unknown")),
            f"{float(result.get('score', 0.0)):.2f}",
            (result.get("reason") or "")[:200],
        )

    llm_label, llm_score, llm_reason = _fmt_row(llm_result)
    clf_label, clf_score, clf_reason = _fmt_row(clf_result)
    table.add_row("模式 A (LLM)", llm_label, llm_score, llm_reason)
    table.add_row("模式 B (分类器)", clf_label, clf_score, clf_reason)

    features_table = Table(title="关键特征", header_style="bold magenta")
    features_table.add_column("特征")
    features_table.add_column("值")

    feature_highlights = {
        "URL 个数": feature_result.text_features.get("url_count", 0),
        "是否包含典型钓鱼词": "是" if feature_result.text_features.get("has_phishing_keywords") else "否",
        "钓鱼关键词命中": ", ".join(feature_result.keyword_hits) if feature_result.keyword_hits else "无",
        "感叹号数量": feature_result.text_features.get("exclamation_count", 0),
        "句子数": feature_result.text_features.get("sentence_count", 0),
    }

    for name, value in feature_highlights.items():
        features_table.add_row(name, str(value))

    console.print(Panel(header, expand=False))
    console.print(table)
    console.print(features_table)


def main() -> None:
    args = parse_args()
    cfg = config.AppConfig()
    console = Console()

    try:
        if not args.email_path.exists():
            raise FileNotFoundError(f"File not found: {args.email_path}")

        if args.email_path.suffix.lower() not in {".eml", ".txt"}:
            raise ValueError("Unsupported file type. Please provide a .eml or .txt file.")

        email_content = load_email(args.email_path)
        basic_features = extract_basic_features(email_content)

        text_features = {
            k: v
            for k, v in basic_features.items()
            if k
            not in {
                "phishing_keyword_hits",
            }
        }
        keyword_hits = basic_features.get("phishing_keyword_hits", [])
        feature_result = FeatureExtractionResult(
            raw_text=email_content.get("body", ""),
            text_features=text_features,
            keyword_hits=keyword_hits,
            llm_features={},
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[red]Failed to process email:[/red] {exc}")
        return

    llm_available = _has_llm_key(cfg)
    if not llm_available and args.mode in {"llm", "hybrid"}:
        console.print(
            "[yellow]OPENAI_API_KEY 未设置，自动降级为分类器模式（模式 B）。[/yellow]"
        )
        args.mode = "clf"

    detection: Dict[str, Any]
    llm_result: Optional[Dict[str, Any]] = None
    clf_result: Optional[Dict[str, Any]] = None
    if args.mode == "llm":
        llm_result = llm_judge(email_content, cfg)
        detection = llm_result
    elif args.mode == "clf":
        from phishing_detector.detectors.classifier import load_or_train_classifier, predict_with_classifier

        classifier = load_or_train_classifier(cfg)
        clf_result = predict_with_classifier(classifier, feature_result)
        detection = clf_result
    else:  # hybrid
        from phishing_detector.detectors.classifier import load_or_train_classifier, predict_with_classifier

        llm_result = llm_judge(email_content, cfg)
        classifier = load_or_train_classifier(cfg)
        clf_result = predict_with_classifier(classifier, feature_result)

        combined_score = (llm_result.get("score", 0.0) + clf_result.get("score", 0.0)) / 2.0
        combined_label = "phishing" if llm_result.get("label") == "phishing" or clf_result.get("label") == "phishing" else "benign"
        combined_reason = (
            f"LLM: {llm_result.get('reason', '')}; CLF: {clf_result.get('reason', '')}"
        ).strip()

        detection = {
            "label": combined_label,
            "score": combined_score,
            "reason": combined_reason,
            "triggered_features": feature_result.keyword_hits,
        }

    if not detection.get("triggered_features"):
        detection["triggered_features"] = feature_result.keyword_hits

    render_output(
        console=console,
        overall=detection,
        llm_result=llm_result,
        clf_result=clf_result,
        feature_result=feature_result,
    )


if __name__ == "__main__":
    main()
