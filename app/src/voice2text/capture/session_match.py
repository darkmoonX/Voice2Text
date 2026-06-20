"""Session matching helpers for app-session capture policies."""
from __future__ import annotations

import re


def extract_process_name(session: object) -> str:
    proc = getattr(session, "Process", None)
    if proc is None:
        return ""
    try:
        return str(proc.name() or "").strip()
    except Exception:
        return ""


def format_session_label(session: object) -> str:
    display_name = str(getattr(session, "DisplayName", "") or "").strip()
    proc_name = extract_process_name(session)
    if display_name.startswith("@") and "audiosrv" in display_name.lower():
        display_name = "System Sounds"
    if display_name and proc_name:
        if display_name.casefold() == proc_name.casefold():
            return proc_name
        return f"{display_name} ({proc_name})"
    if display_name:
        return display_name
    if proc_name:
        return proc_name
    return "System Sounds"


def session_match_tokens(session: object) -> set[str]:
    tokens: set[str] = set()
    label = format_session_label(session).strip().lower()
    if label:
        tokens.add(label)
    display_name = str(getattr(session, "DisplayName", "") or "").strip().lower()
    if display_name:
        tokens.add(display_name)
    proc_name = extract_process_name(session).strip().lower()
    if proc_name:
        tokens.add(proc_name)
        if proc_name.endswith(".exe"):
            tokens.add(proc_name[:-4])
    if not proc_name and (not display_name):
        tokens.add("system sounds")
    return tokens


def token_matches(target: str, tokens: set[str]) -> bool:
    if not target.strip():
        return False
    target_variants = target_match_variants(target)
    token_variants: set[str] = set()
    for token in tokens:
        token_variants.update(target_match_variants(token))
    for needle in target_variants:
        for token in token_variants:
            if not needle or not token:
                continue
            if needle == token:
                return True
            if len(needle) >= 3 and needle in token:
                return True
            if len(token) >= 3 and token in needle:
                return True
    needle_canon = canonicalize_match_text(target)
    if not needle_canon:
        return False
    for token in token_variants:
        token_canon = canonicalize_match_text(token)
        if not token_canon:
            continue
        if needle_canon == token_canon:
            return True
        if len(needle_canon) >= 4 and needle_canon in token_canon:
            return True
        if len(token_canon) >= 4 and token_canon in needle_canon:
            return True
    return False


def target_match_variants(value: str) -> set[str]:
    raw = (value or "").strip().lower()
    if not raw:
        return set()
    variants = {raw}
    if raw.endswith(".exe"):
        variants.add(raw[:-4])
    for inner in re.findall(r"\(([^)]+)\)", raw):
        token = inner.strip().lower()
        if token:
            variants.add(token)
            if token.endswith(".exe"):
                variants.add(token[:-4])
    parts = [part.strip() for part in re.split(r"[|,;/]", raw) if part.strip()]
    variants.update(parts)
    canonical = canonicalize_match_text(raw)
    if canonical:
        variants.add(canonical)
    return {item for item in variants if item}


def canonicalize_match_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return ""
    lowered = lowered.replace(".exe", "")
    lowered = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", lowered)
    return lowered


def session_peak_value(session: object, meter_interface: object | None) -> float:
    if meter_interface is None:
        return 0.0
    try:
        meter = session._ctl.QueryInterface(meter_interface)
        return float(meter.GetPeakValue())
    except Exception:
        return 0.0
