from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

try:
    import argostranslate.package as argos_package
    import argostranslate.translate as argos_translate
except Exception as exc:  # pragma: no cover
    argos_package = None
    argos_translate = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _normalize_code(code: str, default: str) -> str:
    normalized = (code or "").strip().lower()
    if normalized in {"zh-hant", "zh-hans"}:
        return "zh"
    return normalized or default


def _candidate_source_codes(source_code: str, installed_langs) -> list[str]:
    if source_code != "auto":
        return [source_code]

    ordered: list[str] = []
    for code in ("en", "zh", "ja", "ko"):
        if code not in ordered:
            ordered.append(code)

    for lang in installed_langs:
        if lang.code not in ordered:
            ordered.append(lang.code)

    return ordered


def _resolve_translation(source_code: str, target_code: str):
    if argos_translate is None:
        return None, None, None

    installed_langs = argos_translate.get_installed_languages()
    if not installed_langs:
        return None, None, None

    target_lang = next((lang for lang in installed_langs if lang.code == target_code), None)
    if target_lang is None:
        return None, None, None

    for source_candidate in _candidate_source_codes(source_code, installed_langs):
        if source_candidate == target_code:
            continue
        source_lang = next((lang for lang in installed_langs if lang.code == source_candidate), None)
        if source_lang is None:
            continue
        translation = source_lang.get_translation(target_lang)
        if translation is not None:
            return translation, source_lang.code, target_lang.code

    return None, None, target_lang.code


def _try_install(source_code: str, target_code: str) -> tuple[bool, str]:
    if argos_package is None:
        return False, "argostranslate.package unavailable"

    try:
        argos_package.update_package_index()
        available = list(argos_package.get_available_packages())
    except Exception as exc:
        return False, f"Failed to fetch Argos package index: {exc}"

    if not available:
        return False, "No downloadable Argos packages found"

    source_candidates = [source_code] if source_code != "auto" else ["en", "zh", "ja", "ko"]

    for source_candidate in source_candidates:
        if source_candidate == target_code:
            continue
        package = next(
            (
                pkg
                for pkg in available
                if pkg.from_code == source_candidate and pkg.to_code == target_code
            ),
            None,
        )
        if package is None:
            continue

        try:
            download_path = package.download()
            argos_package.install_from_path(download_path)
            return True, f"Installed Argos package: {source_candidate}->{target_code}"
        except Exception as exc:
            return False, f"Failed to install Argos package {source_candidate}->{target_code}: {exc}"

    if source_code == "auto":
        fallback = next((pkg for pkg in available if pkg.to_code == target_code), None)
        if fallback is not None:
            try:
                download_path = fallback.download()
                argos_package.install_from_path(download_path)
                return True, f"Installed Argos package: {fallback.from_code}->{target_code}"
            except Exception as exc:
                return False, f"Failed to install Argos package {fallback.from_code}->{target_code}: {exc}"

    return False, f"No downloadable Argos package for {source_code}->{target_code}"


def _initialize_translation(source_code: str, target_code: str, auto_install: bool):
    if argos_translate is None:
        return None, f"argostranslate unavailable: {_IMPORT_ERROR}", None, None

    if source_code != "auto" and source_code == target_code:
        return None, "Translation source and target are identical", None, None

    translation, source_lang, target_lang = _resolve_translation(source_code, target_code)
    if translation is not None and source_lang and target_lang:
        return translation, f"Translation active: {source_lang}->{target_lang}", source_lang, target_lang

    if not auto_install:
        return None, f"No installed Argos model for {source_code}->{target_code}", source_lang, target_lang

    installed, message = _try_install(source_code, target_code)
    if installed:
        translation, source_lang, target_lang = _resolve_translation(source_code, target_code)
        if translation is not None and source_lang and target_lang:
            return (
                translation,
                f"Translation active: {source_lang}->{target_lang} (auto-installed)",
                source_lang,
                target_lang,
            )

    return None, message, source_lang, target_lang


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argos translate bridge")
    parser.add_argument("--mode", choices=["probe", "translate"], required=True)
    parser.add_argument("--from-code", default="auto")
    parser.add_argument("--to-code", default="zh")
    parser.add_argument("--auto-install", choices=["0", "1"], default="0")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    source_code = _normalize_code(args.from_code, "auto")
    target_code = _normalize_code(args.to_code, "zh")
    auto_install = args.auto_install == "1"

    translation, message, source_lang, target_lang = _initialize_translation(
        source_code, target_code, auto_install
    )

    if args.mode == "probe":
        active = translation is not None
        _emit(
            {
                "active": active,
                "message": message,
                "source": source_lang or source_code,
                "target": target_lang or target_code,
            }
        )
        return 0 if active else 1

    text = sys.stdin.read().strip()
    if not text:
        _emit({"translated": ""})
        return 0

    if translation is None:
        _emit({"translated": "", "error": message})
        return 1

    try:
        translated = translation.translate(text).strip()
    except Exception as exc:
        _emit({"translated": "", "error": f"Translation failed: {exc}"})
        return 1

    _emit({"translated": translated})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
