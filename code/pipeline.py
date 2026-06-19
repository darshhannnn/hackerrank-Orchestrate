"""
pipeline.py -- orchestrates the 2 LLM calls + 1 deterministic call per claim.

process_claim_row() is the single entrypoint used by both run.py (test set)
and evaluation/eval.py (sample set), so the exact same code path is graded
and evaluated.
"""

from pathlib import Path

import prompts
import rules
from llm_client import LLMClient


def _image_id_from_path(path: str) -> str:
    return Path(path).stem


def process_claim_row(
    row: dict,
    llm: LLMClient,
    history_lookup: dict,
    images_root: str,
) -> dict:
    user_id = row["user_id"]
    claim_object = str(row["claim_object"]).strip().lower()
    image_paths_raw = str(row["image_paths"])
    image_paths = [p.strip() for p in image_paths_raw.split(";") if p.strip()]
    image_ids = [_image_id_from_path(p) for p in image_paths]
    full_paths = [str(Path(images_root) / p) for p in image_paths]
    user_claim = str(row["user_claim"])

    # ---- stage 1: claim extraction (text only, no images) ----
    extraction = llm.call_json(
        system=prompts.EXTRACTION_SYSTEM,
        text_prompt=prompts.EXTRACTION_USER_TEMPLATE.format(
            claim_object=claim_object, user_claim=user_claim
        ),
        image_paths=[],
        max_tokens=1200,
    )

    # ---- stage 2: vision grounding (images only, claim text withheld) ----
    vision = llm.call_json(
        system=prompts.VISION_SYSTEM,
        text_prompt=prompts.VISION_USER_TEMPLATE.format(
            claim_object=claim_object,
            claim_summary=extraction.get("claim_summary", ""),
            primary_object_part=extraction.get("primary_object_part", "unknown"),
            primary_issue_type=extraction.get("primary_issue_type", "unknown"),
            n_images=len(image_paths),
            image_ids=", ".join(image_ids),
        ),
        image_paths=full_paths,
        max_tokens=2000,
    )

    # ---- stage 3: deterministic decision (no LLM) ----
    history_row = history_lookup.get(user_id)
    decision = rules.decide(
        claim_object=claim_object,
        image_ids_in_order=image_ids,
        extraction=extraction,
        vision=vision,
        history_row=history_row,
    )

    output_row = {
        "user_id": user_id,
        "image_paths": image_paths_raw,
        "user_claim": user_claim,
        "claim_object": claim_object,
        **decision,
    }
    return output_row
