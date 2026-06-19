"""
Central schema definitions for the Multi-Modal Evidence Review system.
Keeping every allowed value in one place lets the deterministic decision
layer (rules.py) clamp any LLM output to a valid value instead of trusting
free text -- this is what makes the pipeline auditable and grader-safe.
"""

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain",
    "none", "unknown",
}

OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    },
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}


def clamp(value, allowed, default):
    """Return value if it is in the allowed set (case-insensitive), else default."""
    if value is None:
        return default
    v = str(value).strip().lower()
    for a in allowed:
        if v == a:
            return a
    return default


def clamp_object_part(value, claim_object, default="unknown"):
    allowed = OBJECT_PARTS.get(claim_object, {"unknown"})
    return clamp(value, allowed, default)


def normalize_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"true", "yes", "1"}:
        return True
    if v in {"false", "no", "0"}:
        return False
    return default


def join_semicolon(values):
    cleaned = [v.strip() for v in values if v and str(v).strip() and str(v).strip().lower() != "none"]
    # de-dupe while preserving order
    seen = set()
    out = []
    for v in cleaned:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return ";".join(out) if out else "none"


def clamp_risk_flags(values):
    out = []
    for v in values:
        v = str(v).strip().lower()
        if v and v in RISK_FLAGS and v != "none":
            out.append(v)
    return join_semicolon(out)
