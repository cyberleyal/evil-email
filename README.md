# Phishing Detector CLI Prototype

A Python prototype for detecting phishing emails with both LLM-assisted reasoning and traditional machine learning. The project is organized to support feature extraction, LLM scoring, and classifier-based detection from local email files.

## Project layout
```
phishing_detector/
├── main.py                 # CLI entry
├── config.py               # Model/LLM configuration
├── detectors/
│   ├── __init__.py
│   ├── features.py         # Feature extraction
│   ├── llm_judge.py        # LLM scoring
│   └── classifier.py       # Traditional classifier
├── data/
│   └── samples/            # Sample emails
├── models/
│   └── phishing_clf.pkl    # Trained model placeholder
└── README.md
```

## Requirements
- Python 3.10+
- An OpenAI API key (for real LLM scoring)

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The CLI uses `rich` for colored/tabular output and OpenAI for LLM-powered scoring. Configure your API key (required for
`--mode llm` and `--mode hybrid`):
```bash
export OPENAI_API_KEY="sk-..."
```
If `OPENAI_API_KEY` is not set, the CLI will warn and automatically fall back to classifier-only mode.

## Running the CLI
Analyze a local email file (.eml or .txt):
```bash
python phishing_detector/main.py data/samples/sample_email.eml --mode llm
```

- `--mode llm`: LLM-only decision that returns a JSON-parsed label/score/reason.
- `--mode clf`: Traditional classifier using extracted features. A placeholder logistic regression model is trained if no model checkpoint exists.
- `--mode hybrid`: Runs both and reports a combined view (simple OR for label + average score).

Example rich-formatted output (text representation):

```
┌───────────────────────────────┐
│ 检测结果: phishing | 总体风险分数: 0.73 │  (标题以红色突出)
└───────────────────────────────┘

模式结果
┏━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 模式          ┃ Label ┃ Score ┃ Reason                                   ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 模式 A (LLM)  │ phishing │ 0.82 │ 针对密码重置与可疑链接的高风险提示...  │
│ 模式 B (分类器) │ phishing │ 0.64 │                                      │
└──────────────────────────────────────────────────────────────────────────┘

关键特征
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ 特征         ┃ 值                 ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ URL 个数     │ 2                  │
│ 是否包含典型钓鱼词 │ 是               │
│ 钓鱼关键词命中   │ 点击链接, 密码重置 │
│ 感叹号数量     │ 3                  │
│ 句子数         │ 5                  │
└────────────────────────────────────┘
```

## Next steps
- Harden the `llm_judge.py` prompt/formatting and add richer guardrails for malformed outputs.
- Expand `features.py` with domain risk scoring, better parsing of `.eml` attachments, and richer keyword lists.
- Replace the placeholder classifier training with a real dataset and evaluation pipeline.
