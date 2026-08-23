"""JSON-based localization for PlantDiseaseAI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from utils.runtime_paths import get_project_root

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

GROQ_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}


class Translator:
    """Loads translation JSON files and provides keyed lookup with placeholders."""

    def __init__(self, translations_dir: Path | None = None) -> None:
        root = get_project_root()
        self.translations_dir = translations_dir or root / "App" / "resources" / "translations"
        self._locale = "en"
        self._catalog: dict[str, dict[str, Any]] = {}
        self._listeners: list[Callable[[], None]] = []
        self._load_all()

    @property
    def locale(self) -> str:
        return self._locale

    def _load_all(self) -> None:
        for code in SUPPORTED_LANGUAGES:
            path = self.translations_dir / f"{code}.json"
            if not path.exists():
                logger.warning("Missing translation file: %s", path)
                self._catalog[code] = {}
                continue
            with path.open(encoding="utf-8") as handle:
                self._catalog[code] = json.load(handle)

    def on_language_changed(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def set_language(self, code: str) -> None:
        if code not in SUPPORTED_LANGUAGES:
            return
        if code == self._locale:
            return
        self._locale = code
        for callback in self._listeners:
            callback()

    def t(self, key: str, **kwargs: Any) -> str:
        value = self._lookup(self._locale, key)
        if value is None and self._locale != "en":
            value = self._lookup("en", key)
        if value is None:
            return key
        if kwargs:
            try:
                return str(value).format(**kwargs)
            except (KeyError, ValueError):
                return str(value)
        return str(value)

    def translate_class(self, english_name: str) -> str:
        slug_map = {
            "Black Rot": "black_rot",
            "Esca (Black Measles)": "esca",
            "Leaf Blight (Isariopsis Leaf Spot)": "leaf_blight",
            "Healthy": "healthy",
        }
        slug = slug_map.get(english_name)
        if slug:
            translated = self.t(f"classes.{slug}")
            if translated != f"classes.{slug}":
                return translated
        return english_name

    def language_option(self, code: str) -> str:
        return self.t(f"language.options.{code}")

    def groq_language_name(self) -> str:
        return GROQ_LANGUAGE_NAMES.get(self._locale, "English")

    def _lookup(self, locale: str, key: str) -> Any:
        node: Any = self._catalog.get(locale, {})
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node
