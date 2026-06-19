"""
LLM client supporting Ollama models:
  - gemma4:latest via OpenAI API (all text tasks)
  - moondream:v2 via native Ollama API (image description)
  - on-disk JSON caching
  - image downscaling + JPEG re-encoding
"""

import base64
import hashlib
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


VISION_PARSE_SYSTEM = """You are a visual claims inspector. You receive text \
descriptions of images and a customer claim. Based on the descriptions, produce a JSON assessment.

Important rules:
- If the description mentions ANY visible damage on the claimed part, mark claimed_part_visible=true and matches_claimed_part=true
- If the description mentions damage but on a DIFFERENT part, mark matches_claimed_part=false
- If the description says no damage is visible on the relevant part, mark claimed_part_visible=false
- Map description words to allowed values: dent, scratch, crack, broken_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
- For object_part, map to: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, screen, keyboard, trackpad, hinge, lid, corner, port, base, box, package_corner, package_side, seal, label, contents, item, unknown

Respond with ONLY a JSON object:
{"images":[{"image_id":"img_N","object_visible":true,"claimed_part_visible":true,"image_quality_ok":true,"quality_issue":"","visible_issue_type":"dent","visible_object_part":"rear_bumper","matches_claimed_part":true,"authenticity_concern":"none","wrong_object_detected":false,"notes":"short description"}],"overall_evidence_sufficient":true,"overall_reason":"one sentence"}
Include one entry per image in order, using img_1, img_2, etc."""


class LLMClient:
    def __init__(self, config: dict):
        self.config = config
        self.text_model = config["model"]
        self.vision_model = config.get("vision_model", self.text_model)
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.cache_dir = Path(config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._text_client = OpenAI(base_url=self.base_url, api_key="ollama") if OpenAI else None
        self.calls_made = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _prep_image(self, path: str) -> bytes:
        max_dim = self.config.get("image_max_dimension", 512)
        quality = self.config.get("image_jpeg_quality", 70)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def _cache_key(self, system: str, text_prompt: str, image_paths: list[str], model: str) -> str:
        h = hashlib.sha256()
        h.update(system.encode("utf-8"))
        h.update(text_prompt.encode("utf-8"))
        h.update(model.encode("utf-8"))
        for p in sorted(image_paths):
            try:
                h.update(Path(p).read_bytes())
            except FileNotFoundError:
                h.update(p.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _call_text(self, system: str, text_prompt: str, max_tokens: int) -> str:
        resp = self._text_client.chat.completions.create(
            model=self.text_model,
            max_tokens=max_tokens,
            temperature=self.config.get("temperature", 0),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text_prompt},
            ],
        )
        usage = resp.usage
        if usage:
            self.input_tokens += getattr(usage, "prompt_tokens", 0)
            self.output_tokens += getattr(usage, "completion_tokens", 0)
        return resp.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        retry_error_callback=lambda retry_state: retry_state.outcome.result(),
    )
    def _describe_one_image(self, image_path: str, image_id: str) -> str:
        data = self._prep_image(image_path)
        b64 = base64.b64encode(data).decode("utf-8")
        prompt = (
            "What is in this image? Describe the object, the specific part visible, "
            "any damage or issues you can see, and whether the image is clear or blurry."
        )
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.vision_model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [b64]},
                ],
                "stream": False,
                "options": {"num_predict": 300, "temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        if not content:
            content = f"[Image {image_id}: description unavailable]"
        return f"Image {image_id}:\n{content}"

    def _call_vision_describe(self, image_paths: list[str]) -> str:
        descriptions = []
        for i, p in enumerate(image_paths):
            stem = Path(p).stem
            iid = stem if stem.startswith("img_") else f"img_{stem}"
            try:
                desc = self._describe_one_image(p, iid)
                descriptions.append(desc)
            except Exception as e:
                descriptions.append(f"Image {iid}:\n[Error: {e}]")
        return "\n\n".join(descriptions)

    def call_json(
        self,
        system: str,
        text_prompt: str,
        image_paths: list[str] | None = None,
        max_tokens: int = 1000,
    ) -> dict:
        image_paths = image_paths or []
        use_vision = len(image_paths) > 0
        model = self.vision_model if use_vision else self.text_model

        key = self._cache_key(system, text_prompt, image_paths, model)
        cache_file = self._cache_path(key)
        if cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text())

        if use_vision:
            description = self._call_vision_describe(image_paths)
            parsing_prompt = (
                f"Image descriptions from visual inspection:\n{description}\n\n"
                f"Customer claim context:\n{text_prompt}\n\n"
                f"Based on the image descriptions above, produce the required JSON output."
            )
            raw_text = self._call_text(VISION_PARSE_SYSTEM, parsing_prompt, max_tokens)
        else:
            raw_text = self._call_text(system, text_prompt, max_tokens)

        self.calls_made += 1

        try:
            parsed = self._extract_json(raw_text)
        except ValueError:
            print(f"[DEBUG] Raw output ({model}):\n{raw_text[:1000]}")
            raise

        cache_file.write_text(json.dumps(parsed, indent=2))
        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model output: {text[:300]}")
        candidate = text[start:]
        
        # Try parsing as-is first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        
        # Fix incomplete JSON: close open brackets and truncate unterminated strings
        result = []
        in_string = False
        escape = False
        opens = []
        i = 0
        while i < len(candidate):
            ch = candidate[i]
            if escape:
                result.append(ch)
                escape = False
                i += 1
                continue
            if ch == '\\' and in_string:
                result.append(ch)
                escape = True
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                i += 1
                continue
            if in_string:
                result.append(ch)
                i += 1
                continue
            if ch in ('{', '['):
                opens.append(ch)
            elif ch == '}' and opens and opens[-1] == '{':
                opens.pop()
            elif ch == ']' and opens and opens[-1] == '[':
                opens.pop()
            result.append(ch)
            i += 1
        
        fixed = ''.join(result)
        
        # If we ended mid-string, close it
        if in_string:
            fixed += '"'
        
        # Close any remaining open brackets
        for opener in reversed(opens):
            fixed += ']' if opener == '[' else '}'
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot parse JSON: {text[:500]}")

    def usage_summary(self) -> dict:
        return {
            "calls_made": self.calls_made,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
