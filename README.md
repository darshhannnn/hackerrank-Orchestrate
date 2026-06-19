# Multi-Modal Insurance Claim Evidence Review

> **HackerRank challenge submission** — a 3-stage hybrid pipeline that decides
> whether image evidence *supports*, *contradicts*, or is *insufficient* for an
> insurance damage claim, then writes `dataset/output.csv` in the exact required
> 14-column schema.

---

## Repository layout

```
repo_root/                     ← you are here
│
├── README.md                  ← this file
│
├── dataset/                   ← ALL data lives here (images go here too)
│   ├── claims.csv             ← 44 unlabelled test rows  (input to run.py)
│   ├── sample_claims.csv      ← 21 labelled rows         (input to eval.py)
│   ├── user_history.csv       ← per-user prior claim history
│   ├── evidence_requirements.csv  ← evidence checklist per object/type
│   ├── output.csv             ← ⚠ HEADER-ONLY STUB — see §"Generate output.csv"
│   └── images/                ← NOT included (download from repo, see below)
│       ├── sample/
│       │   └── case_NNN/img_N.jpg
│       └── test/
│           └── case_NNN/img_N.jpg
│
└── code_solution/             ← full solution source
    ├── README.md              ← detailed design notes
    ├── config.yaml
    ├── schema.py
    ├── prompts.py
    ├── llm_client.py
    ├── rules.py
    ├── pipeline.py
    ├── run.py
    ├── requirements.txt
    └── evaluation/
        ├── eval.py
        └── evaluation_report.md
```

> **Why `output.csv` is header-only right now**
> The `dataset/images/` folder is not included in the CSV-only zip that was
> available during development.  The pipeline requires real images to call the
> vision LLM.  Once you add the images (step 1 below) and run step 3,
> `output.csv` will be populated with one result row per claim.

---

## Quick start

### 0 — Prerequisites

```bash
python >= 3.9
pip install -r code_solution/requirements.txt
# set your Anthropic key
export ANTHROPIC_API_KEY=sk-ant-...      # Linux / macOS
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
```

### 1 — Add the images

Clone the official repo alongside `dataset/` (or copy the `images/` tree into
`dataset/images/`):

```bash
# option A — git clone then symlink
git clone https://github.com/interviewstreet/hackerrank-orchestrate-june26
cp -r hackerrank-orchestrate-june26/dataset/images dataset/images

# option B — already have the zip
# unzip it so that dataset/images/sample/ and dataset/images/test/ exist
```

### 2 — Run the evaluation (strongly recommended before step 3)

```bash
cd code_solution
python evaluation/eval.py --sample ../dataset/sample_claims.csv
```

This runs the full pipeline on the 21 **labelled** sample rows, compares every
predicted column against the gold labels, and writes:

- **stdout** — per-column accuracy table
- **`evaluation/eval_results.csv`** — one row per claim with every
  expected/predicted pair (open this and eyeball every mismatch before you
  submit)

Tune `rules.py` thresholds or prompt wording in `prompts.py` until you are
happy with accuracy, then proceed to step 3.

### 3 — Generate the real output.csv

```bash
cd code_solution
python run.py --input ../dataset/claims.csv --output ../dataset/output.csv
```

This overwrites the header-only stub with 44 fully-scored result rows.
Re-running is safe: every LLM call is cached on disk by a hash of
`(prompts + image bytes)`, so interrupted runs resume without re-paying for
completed rows, and a `rules.py`-only change costs zero new API calls.

---

## Pipeline design (why 3 stages)

The dataset deliberately includes adversarial rows:

| Trap | Cases |
|------|-------|
| Prompt-injection text in the conversation | case_008, case_036, case_040, case_048, case_055 |
| Rambling conversation — real claim buried at the end | case_006, case_020, case_032 |
| Multi-part claims (two damaged parts in one row) | case_001, case_010, case_019, case_040 |
| Claims in Hindi / Spanish / Hinglish / Chinese-mixed text | case_002, case_016, case_029, case_030, case_046, case_049, case_050 |

A single "throw everything at one LLM call" design is fragile against all of
these simultaneously.  This solution separates concerns so each failure mode
has exactly one place it is handled:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Text extraction  (LLM, text-only)                        │
│  prompts.EXTRACTION_SYSTEM                                           │
│  • reads the full conversation                                       │
│  • ignores instruction-like text, flags it as injection             │
│  • returns: claim_object, object_part, issue_type, summary          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ extracted claim summary (not raw text)
┌────────────────────────────▼────────────────────────────────────────┐
│  Stage 2 — Vision grounding  (LLM, all images in one call)          │
│  prompts.VISION_SYSTEM                                               │
│  • inspects each image independently for visibility / quality        │
│  • claim summary is passed as a *hint of where to look* only —      │
│    the LLM is explicitly instructed NOT to assume it is true         │
│  • returns per-image: visible_part, issue_type, quality_flags,      │
│    matching (bool)                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ structured vision output (JSON)
┌────────────────────────────▼────────────────────────────────────────┐
│  Stage 3 — Deterministic decision layer  (pure Python, no LLM)      │
│  rules.decide()                                                      │
│  • merges vision output + user_history + evidence_requirements       │
│  • computes claim_status purely from image matching — history can    │
│    only ADD risk_flags, never flip the status                        │
│  • schema.py clamps every field to allowed vocabulary                │
└─────────────────────────────────────────────────────────────────────┘
```

### Key design properties

| Property | Where enforced |
|----------|----------------|
| Injection resistance is **structural** — injected text can only ever surface as `risk_flags=text_instruction_present`, never reach the status decision | `pipeline.py` passes only the extracted summary to Stage 2, never the raw `user_claim` |
| History never overrides vision | `rules.decide()` — history lookup only appends to `risk_flags` |
| Allowed-value safety net | `schema.py` — every field clamped before CSV write; stray LLM output degrades to `unknown` / `not_enough_information` |
| Caching & cost control | `llm_client.py` — disk cache keyed by `hash(prompts + image bytes)`, 2 calls/claim regardless of image count |

See `code_solution/README.md` for the full design narrative and
`code_solution/evaluation/evaluation_report.md` for cost / latency numbers.

---

## Output schema

`output.csv` contains exactly these 14 columns in this order:

| Column | Allowed values |
|--------|---------------|
| `user_id` | (from input) |
| `image_paths` | (from input) |
| `user_claim` | (from input) |
| `claim_object` | `car` \| `laptop` \| `package` \| `unknown` |
| `evidence_standard_met` | `true` \| `false` |
| `evidence_standard_met_reason` | free text |
| `risk_flags` | `none` or semicolon-separated flags |
| `issue_type` | `dent` \| `scratch` \| `crack` \| `broken_part` \| `stain` \| `water_damage` \| `torn_packaging` \| `crushed_packaging` \| `missing_contents` \| `unknown` \| `none` |
| `object_part` | part name or `unknown` |
| `claim_status` | `supported` \| `contradicted` \| `not_enough_information` |
| `claim_status_justification` | free text |
| `supporting_image_ids` | `none` or semicolon-separated `img_N` tokens |
| `valid_image` | `true` \| `false` |
| `severity` | `low` \| `medium` \| `high` \| `unknown` \| `none` |

---

## Verified behaviour (no images needed)

The deterministic rules layer was unit-tested with synthetic vision outputs
covering three representative adversarial cases:

| Synthetic case | Mirrors sample row | Expected result |
|----------------|-------------------|-----------------|
| A — claim mismatch + injection attempt | case_005, case_034 | `contradicted`, `text_instruction_present` flag, injection **not** obeyed |
| B — image shows different part, no match | case_006 | `not_enough_information`, `wrong_angle` flag |
| C — clean match | case_001, case_009 | `supported`, severity `medium` |

All three produced the correct schema-valid rows.  Run the synthetic tests at
any time (no API key required):

```bash
cd code_solution
python -c "
import rules, schema
# synthetic test A — mismatch + injection
ext = {'claim_object':'car','object_part':'rear_bumper','issue_type':'dent','injection_detected':True}
vis = {'images':[{'id':'img_1','visible_part':'front_bumper','matching':False,'quality_flags':[],'issue_seen':'scratch'}]}
row = rules.decide(ext, vis, user_id='user_005', history_rows=[])
print('A:', row['claim_status'], '|', row['risk_flags'])
"
```

---

## Known limitation

`output.csv` is header-only because `dataset/images/` was not available at
development time.  **Everything else — prompts, rules, schema, evaluation
harness — is complete and ready to run** the moment you drop the images folder
in place and set `ANTHROPIC_API_KEY`.

If you hit accuracy gaps after running eval (step 2), paste your
`evaluation/eval_results.csv` and the specific mismatch rows for targeted
prompt or rule tuning.
