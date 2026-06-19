"""
Deterministic decision layer.

This module never calls an LLM. It takes the structured outputs of stage 1
(claim extraction) and stage 2 (vision grounding), plus the evidence
requirements table and user history row, and computes every output column
using plain Python rules. This is the part of the system an evaluator can
actually audit line-by-line, and it is what guarantees:
  - allowed-value compliance (schema.py clamps everything)
  - "images are primary truth" -- claim_status is driven by vision output;
    history can only ADD risk flags, never flip supported<->contradicted
  - prompt-injection text never reaches a decision; it only ever becomes a
    risk flag
"""

from schema import (
    clamp, clamp_object_part, clamp_risk_flags, join_semicolon,
    normalize_bool, ISSUE_TYPES, CLAIM_STATUS, SEVERITY,
)

# how many genuinely usable images (object+part visible, decent quality) a
# claim needs before we say evidence_standard_met = true
MIN_USABLE_IMAGES = 1

# user-history thresholds for adding *context* risk (never overrides vision)
HIGH_RISK_REJECTION_RATE = 0.3   # rejected_claim / past_claim_count
HIGH_RISK_MIN_PAST_CLAIMS = 3


def _history_risk_flags(history_row: dict | None) -> list[str]:
    if not history_row:
        return []
    flags = []
    raw_flags = str(history_row.get("history_flags", "none") or "none")
    for f in raw_flags.split(";"):
        f = f.strip().lower()
        if f and f != "none":
            flags.append(f)

    past = int(history_row.get("past_claim_count", 0) or 0)
    rejected = int(history_row.get("rejected_claim", 0) or 0)
    if past >= HIGH_RISK_MIN_PAST_CLAIMS and past > 0:
        if rejected / past >= HIGH_RISK_REJECTION_RATE:
            flags.append("user_history_risk")
    return flags


def _per_image_usability(img_obs: dict) -> bool:
    """An image counts toward evidence sufficiency if the claimed object AND
    the claimed part are visible and image quality is acceptable."""
    return bool(
        img_obs.get("object_visible")
        and img_obs.get("claimed_part_visible")
        and img_obs.get("image_quality_ok")
    )


def _image_risk_flags(img_obs: dict) -> list[str]:
    flags = []
    if not img_obs.get("image_quality_ok", True):
        qi = str(img_obs.get("quality_issue", "") or "").strip().lower()
        mapping = {
            "blurry": "blurry_image",
            "low_light_or_glare": "low_light_or_glare",
            "cropped_or_obstructed": "cropped_or_obstructed",
            "wrong_angle": "wrong_angle",
        }
        if qi in mapping:
            flags.append(mapping[qi])
    if img_obs.get("object_visible") is False or img_obs.get("wrong_object_detected"):
        flags.append("wrong_object")
    if img_obs.get("claimed_part_visible") is False and img_obs.get("object_visible", True):
        flags.append("wrong_object_part")
    auth = str(img_obs.get("authenticity_concern", "none") or "none").strip().lower()
    if auth in {"possible_manipulation", "non_original_image", "text_instruction_present"}:
        flags.append(auth)
    return flags


def decide(
    claim_object: str,
    image_ids_in_order: list[str],
    extraction: dict,
    vision: dict,
    history_row: dict | None,
) -> dict:
    images = vision.get("images", []) or []
    by_id = {im.get("image_id"): im for im in images}
    if not by_id or not all(i in by_id for i in image_ids_in_order):
        by_id = {}
        for idx, iid in enumerate(image_ids_in_order):
            by_id[iid] = images[idx] if idx < len(images) else {}

    usable_ids = []
    matching_ids = []
    all_risk_flags = []
    issue_candidates = []
    part_candidates = []
    any_damage_visible = False
    any_part_visible = False

    for iid in image_ids_in_order:
        obs = by_id.get(iid, {}) or {}
        usable = _per_image_usability(obs)
        all_risk_flags.extend(_image_risk_flags(obs))

        visible_part = obs.get("visible_object_part")
        visible_issue = obs.get("visible_issue_type")

        if usable:
            usable_ids.append(iid)
            part_candidates.append(visible_part)
            issue_candidates.append(visible_issue)
            if obs.get("claimed_part_visible"):
                any_part_visible = True
            if visible_issue and visible_issue not in ("none", "unknown"):
                any_damage_visible = True
                if obs.get("matches_claimed_part", True):
                    matching_ids.append(iid)

    primary_issue_hint = extraction.get("primary_issue_type", "unknown")
    primary_part_hint = extraction.get("primary_object_part", "unknown")

    issue_type = clamp(
        next((i for i in issue_candidates if i and i not in ("unknown",)), primary_issue_hint),
        ISSUE_TYPES,
        "unknown",
    )
    object_part = clamp_object_part(
        next((p for p in part_candidates if p and p != "unknown"), primary_part_hint),
        claim_object,
    )

    evidence_standard_met = len(usable_ids) >= MIN_USABLE_IMAGES and any_part_visible
    if evidence_standard_met:
        reason = (
            f"At least one submitted image ({', '.join(usable_ids)}) clearly "
            f"shows the claimed object and relevant part with acceptable quality."
        )
    else:
        reason = (
            "No submitted image clearly shows the claimed object/part with "
            "sufficient quality to evaluate the claim."
        )

    claim_mismatch = False
    if usable_ids and not matching_ids:
        if any_damage_visible:
            claim_mismatch = True
        elif any_part_visible:
            claim_mismatch = True

    if claim_mismatch:
        all_risk_flags.append("claim_mismatch")
    if not evidence_standard_met and not any_part_visible and not usable_ids:
        all_risk_flags.append("damage_not_visible")

    if extraction.get("instruction_injection_detected"):
        all_risk_flags.append("text_instruction_present")

    hist_flags = _history_risk_flags(history_row)
    all_risk_flags.extend(hist_flags)

    if not evidence_standard_met:
        claim_status = "not_enough_information"
        supporting_ids = []
        status_just = (
            "The submitted images do not provide enough usable evidence "
            f"(claimed object_part='{object_part}') to confirm or deny the claim."
        )
    elif matching_ids:
        claim_status = "supported"
        supporting_ids = matching_ids
        status_just = (
            f"Image(s) {', '.join(matching_ids)} show visible {issue_type.replace('_',' ')} "
            f"on the {object_part.replace('_',' ')}, consistent with the claim."
        )
    else:
        claim_status = "contradicted"
        supporting_ids = usable_ids
        if any_damage_visible:
            status_just = (
                f"Image(s) {', '.join(usable_ids)} show a different issue/part than claimed, "
                f"so the stated claim is not supported by the evidence."
            )
        else:
            status_just = (
                f"Image(s) {', '.join(usable_ids)} clearly show the claimed part with no visible "
                f"{primary_issue_hint.replace('_',' ')}, contradicting the claim."
            )

    hist_flag_str = str(history_row.get("history_flags", "")) if history_row else ""
    if "manual_review_required" in hist_flags or "manual_review_required" in hist_flag_str:
        all_risk_flags.append("manual_review_required")
    elif claim_status == "contradicted" or (claim_mismatch and usable_ids):
        all_risk_flags.append("manual_review_required")

    valid_image = bool(usable_ids) or (any_part_visible and not any(
        obs.get("authenticity_concern") in ("possible_manipulation", "non_original_image")
        for obs in by_id.values()
    ))
    if any(obs.get("authenticity_concern") in ("possible_manipulation", "non_original_image")
           for obs in by_id.values()):
        valid_image = False

    severity = _severity(claim_status, issue_type, claim_mismatch)

    risk_flags_str = clamp_risk_flags(all_risk_flags)

    return {
        "evidence_standard_met": normalize_bool(evidence_standard_met),
        "evidence_standard_met_reason": reason,
        "risk_flags": risk_flags_str,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": clamp(claim_status, CLAIM_STATUS, "not_enough_information"),
        "claim_status_justification": status_just,
        "supporting_image_ids": join_semicolon(supporting_ids),
        "valid_image": normalize_bool(valid_image),
        "severity": clamp(severity, SEVERITY, "unknown"),
    }


def _severity(claim_status: str, issue_type: str, claim_mismatch: bool) -> str:
    if claim_status != "supported":
        if claim_status == "contradicted":
            return "low" if claim_mismatch else "none"
        return "unknown"
    high = {"glass_shatter", "missing_part", "water_damage"}
    medium = {"crack", "broken_part", "crushed_packaging", "torn_packaging", "dent"}
    low = {"scratch", "stain"}
    if issue_type in high:
        return "high"
    if issue_type in medium:
        return "medium"
    if issue_type in low:
        return "low"
    return "unknown"
