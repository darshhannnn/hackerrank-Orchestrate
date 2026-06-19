"""
run.py -- CLI driver.

Usage:
    python run.py --input ../dataset/claims.csv --output ../dataset/output.csv
    python run.py --input ../dataset/sample_claims.csv --output sample_predictions.csv

Reads claims.csv (or sample_claims.csv, ignoring its label columns),
user_history.csv, and evidence_requirements.csv, processes every row through
pipeline.process_claim_row, and writes output.csv with the exact required
schema and column order.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from llm_client import LLMClient
from pipeline import process_claim_row
from schema import OUTPUT_COLUMNS


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_history(path: str) -> dict:
    df = pd.read_csv(path)
    return {row["user_id"]: row.to_dict() for _, row in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to claims.csv or sample_claims.csv")
    ap.add_argument("--output", required=True, help="Path to write output.csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows (debugging)")
    args = ap.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.input).resolve().parent
    images_root = data_dir  # image_paths in CSV are relative to the dataset dir

    history_path = data_dir / config["user_history_file"]
    history_lookup = load_history(str(history_path)) if history_path.exists() else {}

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    llm = LLMClient(config)

    rows_out = []
    start = time.time()
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            result = process_claim_row(row_dict, llm, history_lookup, str(images_root))
        except Exception as e:
            print(f"[row {idx}] FAILED: {e}")
            result = {
                "user_id": row_dict.get("user_id"),
                "image_paths": row_dict.get("image_paths"),
                "user_claim": row_dict.get("user_claim"),
                "claim_object": row_dict.get("claim_object"),
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"Processing error: {e}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "System error during automated review; routed to manual review.",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            }
        rows_out.append(result)
        print(f"[{idx+1}/{len(df)}] {row_dict.get('user_id')} -> "
              f"{result['claim_status']} / {result['issue_type']} / {result['object_part']}")

    elapsed = time.time() - start
    out_df = pd.DataFrame(rows_out)[OUTPUT_COLUMNS]
    out_df.to_csv(args.output, index=False)

    usage = llm.usage_summary()
    print("\n--- run summary ---")
    print(f"rows processed: {len(df)}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"llm calls made: {usage['calls_made']}  cache hits: {usage['cache_hits']}")
    print(f"input tokens: {usage['input_tokens']}  output tokens: {usage['output_tokens']}")
    print(f"output written to: {args.output}")


if __name__ == "__main__":
    main()
