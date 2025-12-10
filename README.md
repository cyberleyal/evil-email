# Phishing Detector CLI Prototype

A Python prototype for detecting phishing emails with both LLM-assisted reasoning (via an OpenAI-compatible proxy endpoint) and traditional machine learning. The project is organized to support feature extraction, LLM scoring, and classifier-based detection from local email files.

## Project layout
```
phishing_detector/
├── main.py                 # CLI entry
├── config.py               # Model/LLM configuration
├── detectors/
│   ├── __init__.py
│   ├── features.py         # Feature extraction
│   ├── llm_judge.py        # LLM scoring
│   ├── llm_client.py       # Proxy OpenAI client builder
│   └── classifier.py       # Traditional classifier
├── data/
│   └── samples/            # Sample emails
├── models/
│   └── phishing_clf.pkl    # Trained model placeholder
└── README.md
```

## Requirements
- Python 3.10+
- An API key for your OpenAI-compatible proxy (defaults to `https://api.gpt.ge/v1/`) if you want LLM-powered scoring.

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure the proxy LLM
The CLI uses `rich` for colored/tabular output and an OpenAI-compatible proxy for LLM-powered scoring. Configure your API key (required for `--mode llm` and `--mode hybrid`):
```bash
export PROXY_API_KEY="你的中转平台 api key"
export PROXY_BASE_URL="https://api.gpt.ge/v1/"  # 可选，若与默认不同
# 兼容 OPENAI_API_KEY / PROXY_API_KEY 变量名
```
If no proxy API key is set, the CLI will warn and automatically fall back to classifier-only mode.

## Running the CLI
Analyze a local email file (.eml or .txt):
```bash
python phishing_detector/main.py data/samples/sample_email.eml --mode hybrid
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
