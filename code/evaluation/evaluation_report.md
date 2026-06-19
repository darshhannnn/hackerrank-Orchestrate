# Operational Analysis

## Architecture

Each claim triggers exactly **3 LLM calls**:

1. **Extraction call** (gemma4:latest, text-only) — reads the conversation, extracts the authoritative claim
2. **Vision description** (moondream:v2, one call per image) — describes each image in plain text
3. **Vision parsing** (gemma4:latest, text-only) — converts moondream descriptions into structured JSON

Stage 3 (`rules.py`) is pure Python, zero LLM calls.

## Call volume

| Dataset | Rows | Images | Vision calls | Text calls | Total |
|---|---|---|---|---|---|
| sample_claims.csv | 20 | 29 | 29 | 40 | 69 |
| claims.csv (test) | 44 | 82 | 82 | 88 | 170 |
| **Total** | 64 | 111 | 111 | 128 | **239** |

## Token usage (approximate)

- Extraction call: ~350 input tokens + ~300 output tokens
- Vision description (per image): ~200 input tokens + ~100 output tokens
- Vision parsing: ~400 input tokens + ~400 output tokens

Estimated totals for the 44-row test set:
- Input tokens: ~52,000
- Output tokens: ~76,000

## Cost

Running on local Ollama models — **$0 API cost**.

Models used:
- gemma4:latest (9.6 GB) — text extraction and vision parsing
- moondream:v2 (1.7 GB) — image description

## Latency / runtime

- Per-row: ~90-100s (extraction ~5s + 1-3 vision descriptions ~10-20s each + parsing ~5s + rules ~instant)
- Full test set (44 rows): ~68 minutes (serial)
- Sample eval (20 rows): ~25 minutes

## Retry strategy

- tenacity exponential backoff, 3 attempts for vision calls
- On-disk JSON cache keyed by content hash — re-runs cost zero additional calls

## Evaluation results (sample_claims.csv)

| Metric | Accuracy |
|---|---|
| evidence_standard_met | 80.0% |
| risk_flags | 25.0% |
| issue_type | 35.0% |
| object_part | 65.0% |
| claim_status | 65.0% |
| supporting_image_ids | 65.0% |
| valid_image | 70.0% |
| severity | 50.0% |
| **Overall** | **56.9%** |

## Known limitations

- moondream:v2 is a 1.7GB vision model with limited accuracy on fine-grained damage classification
- Multi-image handling is weak — moondream sometimes returns minimal descriptions for second+ images
- The two-step vision pipeline (moondream describe → gemma4 parse) introduces information loss
- Upgrading to a larger vision model (llava, cogvlm, or a cloud API) would significantly improve accuracy
