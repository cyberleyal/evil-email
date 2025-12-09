"""Central configuration for models and LLM access."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 256


@dataclass
class ModelConfig:
    model_path: Path = Path("models/phishing_clf.pkl")
    retrain_if_missing: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def resolve_model_path(self) -> Path:
        """Return an absolute path for the classifier checkpoint."""
        return self.model.model_path.resolve()
