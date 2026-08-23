"""Groq LLM disease explanation service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inference.predictor import TopPrediction

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SEC = 45


@dataclass
class GroqExplanation:
    """Structured Groq response for UI rendering."""

    overview: str = ""
    causes: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)
    remedies_organic: list[str] = field(default_factory=list)
    remedies_chemical: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    from_cache: bool = False

    @classmethod
    def from_json(cls, data: dict[str, Any], latency_ms: float = 0.0, from_cache: bool = False) -> GroqExplanation:
        remedies = data.get("remedies") or {}
        if isinstance(remedies, list):
            remedies = {"organic": remedies, "chemical": []}

        return cls(
            overview=str(data.get("overview", "")),
            causes=_as_str_list(data.get("causes")),
            symptoms=_as_str_list(data.get("symptoms")),
            prevention=_as_str_list(data.get("prevention")),
            remedies_organic=_as_str_list(remedies.get("organic")),
            remedies_chemical=_as_str_list(remedies.get("chemical")),
            tips=_as_str_list(data.get("tips")),
            latency_ms=latency_ms,
            from_cache=from_cache,
        )

    @classmethod
    def unavailable(cls) -> GroqExplanation:
        return cls(overview="AI explanation is currently unavailable.")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _image_cache_key(image_path: Path, language: str, crop_slug: str) -> str:
    path = image_path.resolve()
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    return f"{digest}:{language}:{crop_slug}"


def _crop_expertise(crop_name: str, crop_slug: str) -> tuple[str, str, str]:
    slug = crop_slug.lower()
    name = crop_name.lower()
    if "tomato" in slug or "tomato" in name:
        return (
            "tomato farmers",
            "horticulture expert specializing in tomato cultivation",
            "field/greenhouse tomato care practices",
        )
    return (
        "grape farmers",
        "viticulture expert",
        "vineyard/grape care practices",
    )


def _build_prompt(
    crop_name: str,
    crop_slug: str,
    predicted_class: str,
    confidence: float,
    top_predictions: list[TopPrediction],
    language: str,
) -> str:
    top3_lines = "\n".join(
        f"- {p.class_name}: {p.confidence * 100:.2f}%"
        for p in top_predictions[:3]
    )
    is_healthy = predicted_class.strip().lower() == "healthy"
    farmer_label, expert_role, care_context = _crop_expertise(crop_name, crop_slug)

    healthy_instructions = f"""
The prediction is HEALTHY. Return JSON with farmer-friendly content:
- overview: Plant status (no disease detected)
- causes: [] (empty array)
- symptoms: brief note that no disease symptoms were detected
- prevention: preventive care and maintenance tips (3-5 items)
- remedies.organic: general {care_context}
- remedies.chemical: [] (empty — no chemicals needed)
- tips: monitoring advice for ongoing crop health (3-5 items)
""" if is_healthy else """
The prediction is a DISEASE. Return JSON with:
- overview: disease overview (what it is)
- causes: why it happened (humidity, rainfall, fungal infection, etc.)
- symptoms: short bullet list
- prevention: actionable prevention steps
- remedies.organic: organic treatment options
- remedies.chemical: chemical options with safety notes; suggest consulting local experts
- tips: best practices to reduce recurrence
"""

    return f"""You are an expert agricultural advisor helping {farmer_label}.

Language: {language}
IMPORTANT: Return the entire JSON response ONLY in {language}. Do not mix languages.
Use simple, farmer-friendly language that is easy to understand.

Crop: {crop_name}
Predicted condition: {predicted_class}
Confidence: {confidence * 100:.2f}%

Top 3 model predictions:
{top3_lines}

{healthy_instructions}

Rules:
- Be concise and farmer-friendly
- Write ALL string values ONLY in {language}
- Tailor advice specifically to {crop_name} cultivation
- Do NOT recommend unsafe chemical usage
- Suggest consulting local agricultural experts when appropriate
- Return ONLY valid JSON matching this schema (no markdown):

{{
  "overview": "string",
  "causes": ["string"],
  "symptoms": ["string"],
  "prevention": ["string"],
  "remedies": {{
    "organic": ["string"],
    "chemical": ["string"]
  }},
  "tips": ["string"]
}}
"""


class GroqExplanationService:
    """Groq-powered disease explanation with session caching."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.cwd()
        load_dotenv(root / ".env")
        self._api_key = os.getenv("GROQ_API_KEY", "").strip()
        self._cache: dict[str, GroqExplanation] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def get_explanation(
        self,
        *,
        image_path: Path,
        crop_name: str,
        crop_slug: str,
        predicted_class: str,
        confidence: float,
        top_predictions: list[TopPrediction],
        language: str = "en",
    ) -> GroqExplanation:
        """
        Fetch or retrieve cached Groq explanation for a prediction.

        Raises on unrecoverable errors; caller handles UI fallback.
        """
        from desktop_app.i18n import GROQ_LANGUAGE_NAMES

        groq_language = GROQ_LANGUAGE_NAMES.get(language, "English")
        cache_key = _image_cache_key(image_path, language, crop_slug)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.info(
                "Groq cache hit | prediction=%s confidence=%.2f language=%s",
                predicted_class,
                confidence,
                language,
            )
            return GroqExplanation(
                overview=cached.overview,
                causes=list(cached.causes),
                symptoms=list(cached.symptoms),
                prevention=list(cached.prevention),
                remedies_organic=list(cached.remedies_organic),
                remedies_chemical=list(cached.remedies_chemical),
                tips=list(cached.tips),
                latency_ms=cached.latency_ms,
                from_cache=True,
            )

        if not self._api_key:
            logger.error("GROQ_API_KEY not set in .env")
            raise RuntimeError("Groq API key not configured")

        prompt = _build_prompt(
            crop_name, crop_slug, predicted_class, confidence, top_predictions, groq_language
        )
        start = time.perf_counter()

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Groq SDK not installed") from exc

        client = Groq(api_key=self._api_key, timeout=REQUEST_TIMEOUT_SEC)

        logger.info(
            "Groq request | prediction=%s confidence=%.2f crop=%s language=%s",
            predicted_class,
            confidence,
            crop_name,
            language,
        )

        _, expert_role, _ = _crop_expertise(crop_name, crop_slug)

        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a {expert_role}. Respond with valid JSON only. "
                            f"All text values must be in {groq_language} only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            raise RuntimeError(f"Groq API unavailable: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.choices[0].message.content or "{}"

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Invalid Groq JSON response: %s", raw[:200])
            raise RuntimeError("Invalid Groq response format") from exc

        explanation = GroqExplanation.from_json(data, latency_ms=latency_ms)
        self._cache[cache_key] = explanation

        logger.info(
            "Groq success | prediction=%s latency_ms=%.0f language=%s cached=false",
            predicted_class,
            latency_ms,
            language,
        )
        return explanation
