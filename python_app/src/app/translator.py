from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import argostranslate.package as argos_package
    import argostranslate.translate as argos_translate
except Exception:  # pragma: no cover - dependency may be missing during setup
    argos_package = None
    argos_translate = None


@dataclass
class TranslationState:
    active: bool
    message: str


class ArgosTranslator:
    def __init__(
        self,
        enabled: bool,
        source_code: str,
        target_code: str,
        auto_install: bool = True,
    ) -> None:
        self._enabled = enabled
        self._source_code = (source_code or "auto").strip().lower()
        self._target_code = (target_code or "zh").strip().lower()
        self._auto_install = auto_install
        self._translation = None
        self._state = TranslationState(False, "Translation disabled.")
        self._initialize()

    @property
    def state(self) -> TranslationState:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    def translate(self, text: str) -> Optional[str]:
        if not self._enabled or self._translation is None:
            return None
        if not text.strip():
            return None

        try:
            translated = self._translation.translate(text)
        except Exception:
            return None

        translated = translated.strip()
        return translated or None

    def _initialize(self) -> None:
        if not self._enabled:
            self._state = TranslationState(False, "Translation disabled by config.")
            return

        if argos_translate is None:
            self._state = TranslationState(
                False,
                "argostranslate package is unavailable in this environment.",
            )
            return

        if self._source_code == self._target_code:
            self._state = TranslationState(
                False,
                "Translation source and target are identical.",
            )
            return

        translation, source_lang, target_lang = self._resolve_translation()
        if translation is not None and source_lang is not None and target_lang is not None:
            self._translation = translation
            self._state = TranslationState(
                True,
                f"Translation active: {source_lang}->{target_lang}",
            )
            return

        if self._auto_install:
            installed, install_msg = self._try_install_translation_package()
            if installed:
                translation, source_lang, target_lang = self._resolve_translation()
                if translation is not None and source_lang is not None and target_lang is not None:
                    self._translation = translation
                    self._state = TranslationState(
                        True,
                        f"Translation active: {source_lang}->{target_lang} (auto-installed)",
                    )
                    return

            self._state = TranslationState(False, install_msg)
            return

        self._state = TranslationState(
            False,
            f"No installed Argos model for {self._source_code}->{self._target_code}.",
        )

    def _resolve_translation(self):
        if argos_translate is None:
            return None, None, None

        langs = argos_translate.get_installed_languages()
        if not langs:
            return None, None, None

        target = next((l for l in langs if l.code == self._target_code), None)
        if target is None:
            return None, None, None

        candidates = self._candidate_source_codes(langs)
        for src_code in candidates:
            if src_code == target.code:
                continue

            source = next((l for l in langs if l.code == src_code), None)
            if source is None:
                continue

            candidate = source.get_translation(target)
            if candidate is None:
                continue

            return candidate, source.code, target.code

        return None, None, target.code

    def _candidate_source_codes(self, langs) -> list[str]:
        if self._source_code != "auto":
            return [self._source_code]

        ordered: list[str] = []
        for code in ["en", "zh", "ja", "ko"]:
            if code not in ordered:
                ordered.append(code)

        for lang in langs:
            if lang.code not in ordered:
                ordered.append(lang.code)

        return ordered

    def _try_install_translation_package(self) -> tuple[bool, str]:
        if argos_package is None:
            return False, "argostranslate.package is unavailable."

        try:
            argos_package.update_package_index()
            available = list(argos_package.get_available_packages())
        except Exception as exc:
            return False, f"Failed to fetch Argos package index: {exc}"

        if not available:
            return False, "No downloadable Argos packages found."

        if self._source_code == "auto":
            source_candidates = ["en", "zh", "ja", "ko"]
        else:
            source_candidates = [self._source_code]

        for src_code in source_candidates:
            if src_code == self._target_code:
                continue

            package = next(
                (
                    pkg
                    for pkg in available
                    if pkg.from_code == src_code and pkg.to_code == self._target_code
                ),
                None,
            )
            if package is None:
                continue

            try:
                download_path = package.download()
                argos_package.install_from_path(download_path)
                return True, f"Installed Argos package: {src_code}->{self._target_code}"
            except Exception as exc:
                return False, f"Failed to install Argos package {src_code}->{self._target_code}: {exc}"

        if self._source_code == "auto":
            package = next((pkg for pkg in available if pkg.to_code == self._target_code), None)
            if package is not None:
                try:
                    download_path = package.download()
                    argos_package.install_from_path(download_path)
                    return True, (
                        f"Installed Argos package: {package.from_code}->{self._target_code}"
                    )
                except Exception as exc:
                    return False, (
                        f"Failed to install Argos package {package.from_code}->{self._target_code}: {exc}"
                    )

        return False, (
            f"No downloadable Argos package for {self._source_code}->{self._target_code}."
        )