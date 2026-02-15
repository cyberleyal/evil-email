"""Traditional ML classifier for phishing detection."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from phishing_detector.config import AppConfig
from phishing_detector.detectors.features import FeatureExtractionResult


def _feature_vector_from_result(feature_result: FeatureExtractionResult) -> np.ndarray:
    """Project a FeatureExtractionResult into a numeric vector."""

    tf = feature_result.text_features
    vec = [
        tf.get("length", tf.get("text_length", 0.0)),
        tf.get("avg_sentence_length", 0.0),
        tf.get("sentence_count", 0.0),
        tf.get("exclamation_count", 0.0),
        tf.get("has_url", 0.0),
        tf.get("url_count", 0.0),
        tf.get("suspicious_domain_score", 0.0),
        tf.get("phishing_keyword_count", len(feature_result.keyword_hits)),
        feature_result.llm_features.get("llm_like_score", 0.0),
        feature_result.llm_features.get("phishing_risk_score", 0.0),
    ]
    return np.array(vec, dtype=float)


def train_dummy_classifier(samples: List[FeatureExtractionResult] | None = None) -> LogisticRegression:
    """Train a small demo LogisticRegression using heuristic examples."""

    if samples is None:
        samples = [
            FeatureExtractionResult(
                raw_text="",  # benign short note
                text_features={
                    "length": 80,
                    "text_length": 80,
                    "avg_sentence_length": 12,
                    "sentence_count": 3,
                    "exclamation_count": 0,
                    "has_url": 0,
                    "url_count": 0,
                    "suspicious_domain_score": 0.1,
                    "phishing_keyword_count": 0,
                },
                keyword_hits=[],
                llm_features={"llm_like_score": 0.2, "phishing_risk_score": 0.1},
            ),
            FeatureExtractionResult(
                raw_text="",  # suspicious message with links
                text_features={
                    "length": 350,
                    "text_length": 350,
                    "avg_sentence_length": 18,
                    "sentence_count": 5,
                    "exclamation_count": 2,
                    "has_url": 1,
                    "url_count": 3,
                    "suspicious_domain_score": 0.6,
                    "phishing_keyword_count": 2,
                },
                keyword_hits=["密码重置", "点击链接"],
                llm_features={"llm_like_score": 0.7, "phishing_risk_score": 0.8},
            ),
            FeatureExtractionResult(
                raw_text="",  # noisy but benign marketing
                text_features={
                    "length": 250,
                    "text_length": 250,
                    "avg_sentence_length": 16,
                    "sentence_count": 6,
                    "exclamation_count": 1,
                    "has_url": 1,
                    "url_count": 1,
                    "suspicious_domain_score": 0.2,
                    "phishing_keyword_count": 1,
                },
                keyword_hits=["reset password"],
                llm_features={"llm_like_score": 0.5, "phishing_risk_score": 0.3},
            ),
        ]

    X = np.vstack([_feature_vector_from_result(s) for s in samples])
    y = np.array([0, 1, 0])  # simple labels aligned with samples above

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    return clf


def load_or_train_classifier(cfg: AppConfig) -> LogisticRegression:
    """Load classifier from disk or train a dummy model for demonstration."""

    model_path = cfg.resolve_model_path()
    if model_path.exists():
        with model_path.open("rb") as f:
            return pickle.load(f)

    clf = train_dummy_classifier()
    if cfg.model.retrain_if_missing:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as f:
            pickle.dump(clf, f)
    return clf


def predict_with_classifier(classifier: LogisticRegression, feature_result: FeatureExtractionResult) -> Dict[str, object]:
    vec = _feature_vector_from_result(feature_result).reshape(1, -1)
    prob = classifier.predict_proba(vec)[0][1]
    return {
        "label": "phishing" if prob >= 0.5 else "benign",
        "score": float(prob),
        "triggered_features": feature_result.keyword_hits,
    }
