"""
evaluation/eval.py

Runs the pipeline on dataset/sample_claims.csv (which has ground-truth
labels) and reports field-level accuracy against those labels. This is the
required "evaluation workflow" -- run it BEFORE trusting predictions on the
unlabeled claims.csv.

Usage (from code/ directory):
    python evaluation/eval.py --sample ../dataset/sample_claims.csv

It will:
  1. Strip the label columns off sample_claims.csv to build pipeline input
  2. Run the same process_claim_row() used for the real test set
  3. Compare predicted vs. labeled values per column
  4. Print a per-column accuracy report and save it to evaluation/eval_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import LLMClient
from pipeline import process_claim_row
from run import load_config, load_history
from schema import OUTPUT_COLUMNS

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]
LABEL_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in INPUT_COLUMNS]

# columns compared with exact string match; the two justification/reason
# text columns are free-form so we only check they're non-empty and report
# them for manual spot-checking rather than scoring them as pass/fail
EXACT_MATCH_COLUMNS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]
FREE_TEXT_COLUMNS = ["evidence_standard_met_reason", "claim_status_justification"]


def normalize(v):
    return str(v).strip().lower()


def run_eval(sample_path: str, config_path: str, limit: int | None = None):
    config = load_config(config_path)
    data_dir = Path(sample_path).resolve().parent
    history_lookup = load_history(str(data_dir / config["user_history_file"]))

    df = pd.read_csv(sample_path)
    if limit:
        df = df.head(limit)

    llm = LLMClient(config)

    records = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        input_row = {c: row_dict[c] for c in INPUT_COLUMNS}
        pred = process_claim_row(input_row, llm, history_lookup, str(data_dir))
        rec = {"row": idx, "user_id": row_dict["user_id"]}
        for c in LABEL_COLUMNS:
            rec[f"expected_{c}"] = row_dict.get(c)
            rec[f"predicted_{c}"] = pred.get(c)
            if c in EXACT_MATCH_COLUMNS:
                rec[f"match_{c}"] = normalize(row_dict.get(c)) == normalize(pred.get(c))
        records.append(rec)
        print(f"[{idx+1}/{len(df)}] {row_dict['user_id']} "
              f"expected={row_dict.get('claim_status')} predicted={pred.get('claim_status')}")

    results_df = pd.DataFrame(records)
    out_path = Path(__file__).resolve().parent / "eval_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n=== Per-column accuracy ===")
    summary_lines = []
    for c in EXACT_MATCH_COLUMNS:
        col = f"match_{c}"
        acc = results_df[col].mean() if col in results_df else float("nan")
        line = f"{c:30s} {acc*100:5.1f}%  ({int(results_df[col].sum())}/{len(results_df)})"
        print(line)
        summary_lines.append(line)

    overall = results_df[[f"match_{c}" for c in EXACT_MATCH_COLUMNS]].mean(axis=1)
    print(f"\nOverall mean field accuracy: {overall.mean()*100:.1f}%")
    print(f"Full results written to: {out_path}")

    usage = llm.usage_summary()
    print(f"\nLLM calls: {usage['calls_made']}  cache hits: {usage['cache_hits']}  "
          f"input_tokens: {usage['input_tokens']}  output_tokens: {usage['output_tokens']}")

    return results_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="../dataset/sample_claims.csv")
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_eval(args.sample, args.config, args.limit)
