"""
code/evaluation/main.py — Evaluation entry point.

Usage:
    python code/evaluation/main.py --sample dataset/sample_claims.csv
"""
if __name__ == "__main__":
    import eval as eval_module
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="../dataset/sample_claims.csv")
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    eval_module.run_eval(args.sample, args.config, args.limit)
