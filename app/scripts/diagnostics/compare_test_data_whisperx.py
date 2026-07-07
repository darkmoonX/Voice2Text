"""Compare WhisperX direct transcript vs project incremental realtime flow on test_data audio."""
from __future__ import annotations

import argparse
from datetime import datetime
import difflib
import gc
import html as html_lib
import json
import os
import re
import shutil
import sys
import threading
import time
import warnings
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
REPO_ROOT = APP_ROOT.parent
COMPARE_ROOT = SRC_ROOT / "tests" / "compare_whisperx_test"
DEFAULT_INPUT_DIR = COMPARE_ROOT / "input"
DEFAULT_OUTPUT_ROOT = COMPARE_ROOT / "output"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.capture import build_capture_from_config
from voice2text.config import RuntimeConfig
from voice2text.pipeline.direct_transcription import (
    audio_duration_seconds as _audio_duration_seconds,
    call_speaker_profile_reconcile as _call_speaker_profile_reconcile,
    decode_to_wav_16k_mono as _decode_to_wav_16k_mono,
    read_wav as _read_wav,
    run_direct_transcription,
)
from voice2text.pipeline.gpu_telemetry import GpuTelemetryReporter
from voice2text.pipeline.segment_artifacts import SegmentArtifacts
from voice2text.pipeline.subtitle_assembler import SubtitleAssembler
from voice2text.pipeline.text_delta_logger import TextDeltaLogger
from voice2text.pipeline.transcription_loop import TranscriptionLoopDeps, TranscriptionLoopEngine
from voice2text.pipeline.transcript_exporter import TranscriptExportOptions, TranscriptExporterSession
from voice2text.settings_persistence import apply_updates_to_config, load_persisted_updates
from voice2text.stt.factory import create_stt_transcriber
from voice2text.stt.preprocessing import create_audio_preprocessing_pipeline

try:
    # Reuse the project OpenCC wrapper (cached). `hans` folds Traditional -> Simplified.
    from voice2text.stt.audio_utils import normalize_chinese_script as _normalize_chinese_script
except Exception:  # pragma: no cover - OpenCC/audio_utils unavailable
    def _normalize_chinese_script(text: str, script: str | None) -> str:
        return text


def _fold_cjk_script(text: str) -> str:
    """Fold Simplified/Traditional to one script (Simplified) for script-insensitive compare.

    WhisperX alternates 简/繁 for the same phrase, which otherwise inflates the diff and CER.
    ASCII/English are untouched; only Traditional Han characters are mapped to Simplified.
    """
    if not text:
        return text
    try:
        return _normalize_chinese_script(text, "hans")
    except Exception:  # pragma: no cover
        return text


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"(?s).*torchcodec is not installed correctly.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
    message=r".*torchcodec.*",
)


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def _install_run_logger(output_root: Path):
    log_path = output_root / "compare_run.log"
    log_file = log_path.open("a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _TeeStream(old_stdout, log_file)
    sys.stderr = _TeeStream(old_stderr, log_file)

    def _restore() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()

    return log_path, _restore


def _format_memory_snapshot() -> str:
    rss_mb = -1.0
    cuda_alloc_mb = -1.0
    cuda_reserved_mb = -1.0
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        try:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
                get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
                get_mem.restype = wintypes.BOOL
                get_current = ctypes.windll.kernel32.GetCurrentProcess
                get_current.restype = wintypes.HANDLE
                if get_mem(
                    get_current(),
                    ctypes.byref(counters),
                    counters.cb,
                ):
                    rss_mb = float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass
    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                cuda_alloc_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            except Exception:
                pass
            try:
                cuda_reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
            except Exception:
                pass
    except Exception:
        pass
    return f"rss={rss_mb:.1f}MB; cuda_alloc={cuda_alloc_mb:.1f}MB; cuda_reserved={cuda_reserved_mb:.1f}MB"


def _release_runtime_memory(tag: str) -> None:
    print(f"[mem] {tag} before: {_format_memory_snapshot()}", flush=True)
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    print(f"[mem] {tag} after : {_format_memory_snapshot()}", flush=True)


def _dispose_transcriber(transcriber: object | None) -> None:
    if transcriber is None:
        return
    # Best-effort teardown of heavy WhisperX objects so GC can reclaim memory early.
    for attr in (
        "_model",
        "_diarization_pipeline",
        "_speaker_embedding_inference",
        "_speaker_identity_engine",
    ):
        try:
            if hasattr(transcriber, attr):
                setattr(transcriber, attr, None)
        except Exception:
            pass
    try:
        cache = getattr(transcriber, "_align_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        setattr(transcriber, "_align_cache", {})
    except Exception:
        pass
    try:
        setattr(transcriber, "_last_transcription_meta", {})
    except Exception:
        pass


def _strip_reference_annotations(text: str) -> str:
    """Strip ground-truth subtitle annotations the ASR can never produce.

    Round 0002-#2: the GT subtitles carry translator glosses in (full/half-width) parentheses
    (`他（导游）已经` -> `他已经`, `Alcantara（汽车内饰材料）` -> `Alcantara`) and occasional CJK
    speaker-label prefixes (`本地向导：…`). Verbatim ASR can't emit these, so they inflate
    missing/extra and the CER. Stripping them makes ref_cer reflect ASR quality, not annotation
    mismatch. Applied symmetrically to reference + candidate; candidates almost never contain these
    patterns (ASR rarely emits full-width parens/colons), so it is reference-only in effect.
    """
    cleaned = re.sub(r"（[^（）]*）", "", text)
    cleaned = re.sub(r"\([^()]*\)", "", cleaned)
    # CJK speaker-label prefix at line start, full-width colon only (avoid touching legitimate
    # half-width colons in candidate text); short token so dialogue lines are not over-stripped.
    cleaned = re.sub(r"(?m)^\s*[^\s：（）]{1,8}：\s*", "", cleaned)
    return cleaned


def _normalize_for_compare(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"(?m)^\s*S\d+:\s*", "", _strip_reference_annotations(text))
    cleaned = re.sub(r"(?m)^\s*>>\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\[spk_\d+\]\s*", " ", cleaned)
    cleaned = re.sub(r"\bS\d+:\s*", " ", cleaned)
    cleaned = re.sub(r"\s*>>\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return _fold_cjk_script(cleaned)


def _normalize_for_html_compare(text: str) -> str:
    if not text:
        return ""
    cleaned = _strip_reference_annotations(str(text))

    def _speaker_repl(match: re.Match[str]) -> str:
        speaker = _normalize_speaker_token(match.group(1))
        return f"[{speaker.lower()}] " if speaker else ""

    cleaned = re.sub(r"(?i)\[(spk_\d+|speaker_\d+|s\d+)\]\s*", _speaker_repl, cleaned)
    cleaned = re.sub(r"(?im)^\s*(S\d+|SPK_\d+|SPEAKER_\d+):\s*", _speaker_repl, cleaned)
    cleaned = re.sub(r"(?im)^\s*>>\s*", "", cleaned)
    cleaned = re.sub(r"\b(SPK_\d+|SPEAKER_\d+):\s*", _speaker_repl, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*>>\s*", " ", cleaned)
    rows = []
    for raw in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip().lower()
        if line:
            rows.append(line)
    return _fold_cjk_script("\n".join(rows).strip())


def _normalize_incremental_text(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for raw in str(text).splitlines():
        cleaned = re.sub(r"[ \t]+", " ", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def _format_progress_bar(pct: float, *, width: int = 28) -> str:
    bounded = max(0.0, min(1.0, float(pct)))
    filled = int(round(bounded * float(width)))
    return "[" + ("#" * filled) + ("-" * max(0, width - filled)) + "]"


def _format_runtime_timing(timing: object) -> str:
    if not isinstance(timing, dict):
        return ""
    return (
        f"raw={float(timing.get('raw_artifact_seconds') or 0.0):.3f}s "
        f"preprocess={float(timing.get('preprocess_seconds') or 0.0):.3f}s "
        f"stt_artifact={float(timing.get('stt_artifact_seconds') or 0.0):.3f}s "
        f"transcribe={float(timing.get('transcribe_seconds') or 0.0):.3f}s "
        f"language={float(timing.get('language_route_seconds') or 0.0):.3f}s "
        f"timestamp={float(timing.get('timestamp_enrich_seconds') or 0.0):.3f}s "
        f"merge={float(timing.get('merge_seconds') or 0.0):.3f}s "
        f"payload={float(timing.get('subtitle_payload_seconds') or 0.0):.3f}s "
        f"total={float(timing.get('window_total_seconds') or 0.0):.3f}s"
    )


def _append_runtime_snapshot(base: str, incoming: str) -> str:
    left = _normalize_incremental_text(base)
    right = _normalize_incremental_text(incoming)
    if not left:
        return right
    if not right or right in left:
        return left
    max_len = min(len(left), len(right), 2000)
    for size in range(max_len, 0, -1):
        if left.endswith(right[:size]):
            return _normalize_incremental_text(left + right[size:])
    sep = "\n" if re.match(r"^\s*(?:>>|S\d+:)\s*", right) else " "
    return _normalize_incremental_text(left + sep + right)


def _is_timestamp_token(token: str) -> bool:
    raw = str(token or "").strip()
    if not raw:
        return False
    return bool(re.fullmatch(r"[0-9:\-.,>\s]+", raw))


def _normalize_speaker_token(token: str) -> str:
    raw = str(token or "").strip()
    if not raw:
        return ""
    m = re.fullmatch(r"(?i)spk_(\d+)", raw)
    if m is not None:
        return f"SPK_{int(m.group(1)):03d}"
    m = re.fullmatch(r"(?i)speaker_(\d+)", raw)
    if m is not None:
        return f"SPK_{int(m.group(1)):03d}"
    m = re.fullmatch(r"(?i)s(\d+)", raw)
    if m is not None:
        return f"SPK_{int(m.group(1)):03d}"
    return ""


def _strip_line_prefix_metadata(line: str) -> str:
    text = str(line or "").strip()
    while text.startswith("["):
        close = text.find("]")
        if close <= 1:
            break
        token = text[1:close].strip()
        if _normalize_speaker_token(token) or _is_timestamp_token(token):
            text = text[close + 1 :].strip()
            continue
        break
    text = re.sub(r"^\s*S\d+:\s*", "", text)
    return text.strip()


def _extract_project_txt_speaker_and_body(line: str) -> tuple[str, str]:
    text = str(line or "").strip()
    speaker = ""
    while text.startswith("["):
        close = text.find("]")
        if close <= 1:
            break
        token = text[1:close].strip()
        if _is_timestamp_token(token):
            text = text[close + 1 :].strip()
            continue
        normalized_speaker = _normalize_speaker_token(token)
        if normalized_speaker:
            speaker = normalized_speaker
            text = text[close + 1 :].strip()
            continue
        break
    m = re.match(r"^\s*((?:S\d+)|(?:SPK_\d+)|(?:SPEAKER_\d+)):\s*(.*)$", text, flags=re.IGNORECASE)
    if m is not None:
        speaker = _normalize_speaker_token(m.group(1))
        text = m.group(2).strip()
    text = re.sub(r"^\s*>>\s*", "", text).strip()
    return (speaker, text)


def _project_txt_to_single_line(text: str) -> str:
    if not text:
        return ""
    rows: list[str] = []
    last_speaker = ""
    for raw in str(text).splitlines():
        speaker, line = _extract_project_txt_speaker_and_body(raw)
        if not line:
            continue
        if speaker and speaker != last_speaker:
            rows.append(f"[{speaker.lower()}]")
            last_speaker = speaker
        rows.append(line)
    return re.sub(r"\s+", " ", " ".join(rows)).strip()


_SPEAKER_MARKER_BREAK_RE = re.compile(r"\s*(\[(?:spk_[0-9A-Za-z]+|s[0-9]+|speaker[_0-9A-Za-z]*)\])", re.IGNORECASE)


def _break_on_speaker_switch(text: str) -> str:
    """Put each speaker turn on its own line for the human-readable *_for_compare.txt files.

    Inserts a newline before every inline speaker marker so a switch is visible at a
    glance (easy to eyeball against an input spk_subtitles reference). This only changes
    the written .txt layout — CER and speaker-sequence metrics are computed from the
    single-line `*_text_for_compare` strings elsewhere, so they are unaffected.
    """
    if not text:
        return text
    return _SPEAKER_MARKER_BREAK_RE.sub(lambda m: "\n" + m.group(1), str(text)).lstrip("\n")


def _speaker_accuracy_vs_truth(case_dir: Path, input_path: str) -> dict | None:
    """Score direct/realtime speaker attribution against a ground-truth spk_subtitles.

    Ground truth is `<input-clip-dir>/spk_subtitles` (sibling of the source audio).
    Reuses the standalone spk_accuracy_vs_truth scorer so the numbers match that tool
    exactly. Returns None when no ground truth exists; never raises into the run.
    """
    try:
        ref_path = Path(input_path).parent / "spk_subtitles"
        if not ref_path.exists():
            return None
        import sys as _sys
        _here = str(Path(__file__).resolve().parent)
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from spk_accuracy_vs_truth import (
            detect_unit,
            ref_tokens_from_file,
            cand_tokens_direct,
            cand_tokens_realtime,
            score_attribution,
        )

        unit = detect_unit(ref_path.read_text(encoding="utf-8"))
        ref = ref_tokens_from_file(ref_path, unit)
        out: dict = {"unit": unit, "reference": str(ref_path)}
        dj = Path(case_dir) / "direct_whisperx.json"
        rj = Path(case_dir) / "realtime_project.json"
        if dj.exists():
            d = score_attribution(cand_tokens_direct(dj, unit), ref)
            out.update(
                ref_speakers=int(d["ref_speakers"]),
                direct_pred_speakers=int(d["pred_speakers"]),
                direct_speaker_accuracy=float(d["speaker_accuracy"]),
                direct_aligned_tokens=int(d["aligned_tokens"]),
            )
        if rj.exists():
            r = score_attribution(cand_tokens_realtime(rj, unit), ref)
            out.setdefault("ref_speakers", int(r["ref_speakers"]))
            out.update(
                realtime_pred_speakers=int(r["pred_speakers"]),
                realtime_speaker_accuracy=float(r["speaker_accuracy"]),
                realtime_aligned_tokens=int(r["aligned_tokens"]),
            )
        return out
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the run
        return {"error": str(exc)}


def _project_txt_to_compare_lines(text: str, *, include_speaker: bool = False) -> str:
    if not text:
        return ""
    rows: list[str] = []
    for raw in str(text).splitlines():
        speaker, line = _extract_project_txt_speaker_and_body(raw)
        if line:
            if include_speaker and speaker:
                rows.append(f"[{speaker.lower()}] {line}")
                continue
            rows.append(line)
    return "\n".join(rows).strip()


def _speaker_marker_sequence(compare_text: str) -> list[str]:
    return [m.group(1).lower() for m in re.finditer(r"\[(spk_\d+)\]", str(compare_text or ""), flags=re.IGNORECASE)]


_SPEAKER_LABEL_RE = re.compile(
    r"\[(spk_\d+|speaker_\d+|s\d+)\]|(?<![\w\[])(spk_\d+|speaker_\d+|s\d+):",
    flags=re.IGNORECASE,
)


def _normalize_speaker_label(label: object) -> str:
    src = str(label or "").strip()
    match = re.search(r"(\d+)", src)
    if match is None:
        return ""
    return f"spk_{int(match.group(1)):03d}"


def _cosine_similarity(left: list[object], right: list[object]) -> float:
    try:
        import numpy as np

        a = np.asarray(left, dtype=np.float32).reshape(-1)
        b = np.asarray(right, dtype=np.float32).reshape(-1)
        if a.size == 0 or a.size != b.size:
            return -1.0
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an <= 1e-8 or bn <= 1e-8:
            return -1.0
        return float(np.dot(a / an, b / bn))
    except Exception:
        return -1.0


def _speaker_profile_duration_remap(profile_path: str | Path, *, max_speakers: int) -> dict[str, str]:
    """Map noisy profile IDs to the nearest dominant profile IDs for display/export."""
    limit = int(max(0, max_speakers))
    if limit <= 0:
        return {}
    path = Path(profile_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(raw_profiles, list):
        return {}
    profiles: list[dict[str, object]] = []
    for row in raw_profiles:
        if not isinstance(row, dict):
            continue
        label = _normalize_speaker_label(row.get("id"))
        centroid = row.get("centroid")
        if not label or not isinstance(centroid, list):
            continue
        profiles.append(
            {
                "label": label,
                "centroid": centroid,
                "total_seconds": float(row.get("total_seconds", 0.0) or 0.0),
            }
        )
    if len(profiles) <= limit:
        return {}

    def _label_index(label: str) -> int:
        match = re.search(r"(\d+)", label)
        return int(match.group(1)) if match else 999999

    dominant = sorted(
        profiles,
        key=lambda item: (-float(item.get("total_seconds", 0.0) or 0.0), _label_index(str(item.get("label") or ""))),
    )[:limit]
    dominant_labels = {str(item.get("label") or "") for item in dominant}
    fallback_label = str(dominant[0].get("label") or "") if dominant else ""
    remap: dict[str, str] = {}
    for profile in profiles:
        label = str(profile.get("label") or "")
        if not label:
            continue
        if label in dominant_labels:
            remap[label] = label
            continue
        best_label = fallback_label
        best_similarity = -1.0
        for candidate in dominant:
            similarity = _cosine_similarity(
                profile.get("centroid") if isinstance(profile.get("centroid"), list) else [],
                candidate.get("centroid") if isinstance(candidate.get("centroid"), list) else [],
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = str(candidate.get("label") or fallback_label)
        if best_label:
            remap[label] = best_label
    return remap


def _speaker_profile_diagnostics(profile_path: str | Path) -> dict[str, object]:
    path = Path(profile_path)
    if not path.exists():
        return {"path": str(path), "profile_count": 0, "profiles": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "profile_count": 0, "profiles": [], "error": str(exc)}
    raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(raw_profiles, list):
        return {"path": str(path), "profile_count": 0, "profiles": []}

    profiles: list[dict[str, object]] = []
    for row in raw_profiles:
        if not isinstance(row, dict):
            continue
        label = _normalize_speaker_label(row.get("id"))
        if not label:
            continue
        exemplars = row.get("exemplars")
        profiles.append(
            {
                "label": label,
                "total_seconds": round(float(row.get("total_seconds", 0.0) or 0.0), 3),
                "samples": int(row.get("samples", 0) or 0),
                "weight": round(float(row.get("weight", 0.0) or 0.0), 3),
                "observed_labels": list(row.get("observed_labels") or []),
                "exemplar_count": len(exemplars) if isinstance(exemplars, list) else 1,
            }
        )
    profiles.sort(key=lambda item: (-float(item.get("total_seconds", 0.0) or 0.0), str(item.get("label") or "")))
    return {"path": str(path), "profile_count": len(profiles), "profiles": profiles}


def _format_speaker_profile_diagnostics(diag: dict[str, object], *, limit: int = 12) -> list[str]:
    rows = [
        f"path={diag.get('path', '')}",
        f"profile_count={diag.get('profile_count', 0)}",
    ]
    profiles = diag.get("profiles")
    if not isinstance(profiles, list):
        return rows
    for profile in profiles[: max(0, int(limit))]:
        if not isinstance(profile, dict):
            continue
        labels = ",".join(str(item) for item in list(profile.get("observed_labels") or []))
        rows.append(
            f"{profile.get('label', '')}: "
            f"seconds={float(profile.get('total_seconds', 0.0) or 0.0):.3f}; "
            f"samples={int(profile.get('samples', 0) or 0)}; "
            f"weight={float(profile.get('weight', 0.0) or 0.0):.3f}; "
            f"exemplars={int(profile.get('exemplar_count', 1) or 1)}; "
            f"observed={labels}"
        )
    return rows


def _rewrite_speaker_labels_text(text: str, *, profile_remap: dict[str, str] | None = None) -> str:
    """Normalize speaker labels by first visible occurrence after optional profile collapse."""
    src = str(text or "")
    collapse = {str(k).lower(): str(v).lower() for (k, v) in (profile_remap or {}).items()}
    display_map: dict[str, str] = {}

    def _display_label(label: str) -> str:
        normalized = _normalize_speaker_label(label)
        if not normalized:
            return ""
        collapsed = collapse.get(normalized, normalized)
        existing = display_map.get(collapsed)
        if existing:
            return existing
        display = f"spk_{len(display_map):03d}"
        display_map[collapsed] = display
        return display

    def _replace(match: re.Match[str]) -> str:
        label = match.group(1) or match.group(2) or ""
        display = _display_label(label)
        if not display:
            return match.group(0)
        return f"[{display}]"

    return _SPEAKER_LABEL_RE.sub(_replace, src).strip()


def _rewrite_json_speaker_labels(payload: object, *, profile_remap: dict[str, str] | None = None) -> object:
    display_map: dict[str, str] = {}
    collapse = {str(k).lower(): str(v).lower() for (k, v) in (profile_remap or {}).items()}

    def _display(label: object) -> str:
        normalized = _normalize_speaker_label(label)
        if not normalized:
            return str(label or "")
        collapsed = collapse.get(normalized, normalized)
        if collapsed not in display_map:
            display_map[collapsed] = f"spk_{len(display_map):03d}"
        return display_map[collapsed]

    def _walk(value: object) -> object:
        if isinstance(value, dict):
            out: dict[str, object] = {}
            for key, item in value.items():
                if str(key) in {"speaker", "profile_speaker"}:
                    out[str(key)] = _display(item)
                else:
                    out[str(key)] = _walk(item)
            return out
        if isinstance(value, list):
            return [_walk(item) for item in value]
        if isinstance(value, str):
            return _rewrite_speaker_labels_text(value, profile_remap=profile_remap)
        return value

    return _walk(payload)


def _parse_export_time_txt(value: str) -> float:
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$", str(value or "").strip())
    if match is None:
        return -1.0
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return float((hours * 3600) + (minutes * 60) + seconds) + (float(millis) / 1000.0)


def _parse_export_time_srt(value: str) -> float:
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", str(value or "").strip())
    if match is None:
        return -1.0
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return float((hours * 3600) + (minutes * 60) + seconds) + (float(millis) / 1000.0)


def _format_export_time_txt(seconds: float) -> str:
    millis = max(0, int(round(float(seconds) * 1000.0)))
    hours = millis // 3600000
    millis -= hours * 3600000
    minutes = millis // 60000
    millis -= minutes * 60000
    secs = millis // 1000
    millis -= secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_export_time_srt(seconds: float) -> str:
    millis = max(0, int(round(float(seconds) * 1000.0)))
    hours = millis // 3600000
    millis -= hours * 3600000
    minutes = millis // 60000
    millis -= minutes * 60000
    secs = millis // 1000
    millis -= secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _join_export_cue_text(left: str, right: str) -> str:
    first = str(left or "").strip()
    second = str(right or "").strip()
    if not first:
        return second
    if not second:
        return first
    if re.fullmatch(r"[\.,!?;:，。！？；：、)\]\}】》」』]", second[:1]):
        return first + second
    return f"{first} {second}"


def _coalesce_txt_same_speaker_cues(text: str, *, max_gap_seconds: float = 2.0) -> str:
    """Merge adjacent TXT cues that became same-speaker after profile remapping."""
    cue_re = re.compile(
        r"^\[(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*->\s*"
        r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\]\s*"
        r"(?:\[(?P<speaker>spk_\d+)\]\s*)?(?P<body>.*)$",
        flags=re.IGNORECASE,
    )
    rows: list[dict[str, object]] = []
    passthrough: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = cue_re.match(line)
        if match is None:
            passthrough.append(line)
            continue
        start = _parse_export_time_txt(match.group("start"))
        end = _parse_export_time_txt(match.group("end"))
        body = str(match.group("body") or "").strip()
        if start < 0.0 or end < start or not body:
            passthrough.append(line)
            continue
        speaker = _normalize_speaker_label(match.group("speaker"))
        if rows:
            prev = rows[-1]
            same_speaker = str(prev.get("speaker") or "") == speaker
            gap = start - float(prev.get("end", start) or start)
            prev_body = str(prev.get("body") or "")
            if (
                same_speaker
                and gap <= float(max_gap_seconds)
            ):
                prev["end"] = float(max(float(prev.get("end", end) or end), end))
                prev["body"] = _join_export_cue_text(prev_body, body)
                continue
        rows.append({"start": float(start), "end": float(end), "speaker": speaker, "body": body})
    rendered: list[str] = []
    for row in rows:
        speaker = str(row.get("speaker") or "")
        prefix = f"[{speaker}] " if speaker else ""
        rendered.append(
            f"[{_format_export_time_txt(float(row.get('start') or 0.0))} -> "
            f"{_format_export_time_txt(float(row.get('end') or 0.0))}] "
            f"{prefix}{str(row.get('body') or '').strip()}".strip()
        )
    rendered.extend(passthrough)
    return "\n".join(rendered).strip()


def _should_merge_export_cue(prev: dict[str, object], row: dict[str, object], *, max_gap_seconds: float) -> bool:
    if str(prev.get("speaker") or "") != str(row.get("speaker") or ""):
        return False
    gap = float(row.get("start", 0.0) or 0.0) - float(prev.get("end", 0.0) or 0.0)
    if gap > float(max_gap_seconds):
        return False
    return True


def _coalesce_srt_same_speaker_cues(text: str, *, max_gap_seconds: float = 2.0) -> str:
    blocks = re.split(r"\r?\n\s*\r?\n", str(text or "").strip())
    rows: list[dict[str, object]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing = lines[1] if re.fullmatch(r"\d+", lines[0]) and len(lines) >= 3 else lines[0]
        text_lines = lines[2:] if timing == lines[1] else lines[1:]
        match = re.match(
            r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
            r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})$",
            timing,
        )
        if match is None:
            continue
        body_line = " ".join(text_lines).strip()
        speaker, body = _parse_export_speaker_prefixed_text(body_line)
        start = _parse_export_time_srt(match.group("start"))
        end = _parse_export_time_srt(match.group("end"))
        if start < 0.0 or end < start or not body:
            continue
        row = {"start": float(start), "end": float(end), "speaker": speaker, "body": body}
        if rows and _should_merge_export_cue(rows[-1], row, max_gap_seconds=max_gap_seconds):
            rows[-1]["end"] = float(max(float(rows[-1].get("end", end) or end), end))
            rows[-1]["body"] = _join_export_cue_text(str(rows[-1].get("body") or ""), body)
            continue
        rows.append(row)
    rendered: list[str] = []
    for index, row in enumerate(rows, start=1):
        speaker = str(row.get("speaker") or "")
        prefix = f"[{speaker}] " if speaker else ""
        rendered.extend(
            [
                str(index),
                f"{_format_export_time_srt(float(row.get('start') or 0.0))} --> "
                f"{_format_export_time_srt(float(row.get('end') or 0.0))}",
                f"{prefix}{str(row.get('body') or '').strip()}".strip(),
                "",
            ]
        )
    return "\n".join(rendered).strip()


def _parse_export_speaker_prefixed_text(text: str) -> tuple[str, str]:
    body = str(text or "").strip()
    match = re.match(r"^\[(spk_\d+|speaker_\d+|s\d+)\]\s*(.+)$", body, flags=re.IGNORECASE)
    if match is not None:
        return (_normalize_speaker_label(match.group(1)), str(match.group(2)).strip())
    return ("", body)


def _coalesce_json_same_speaker_cues(payload: object, *, max_gap_seconds: float = 2.0) -> object:
    if not isinstance(payload, dict):
        return payload
    raw_cues = payload.get("cues")
    if not isinstance(raw_cues, list):
        return payload
    rows: list[dict[str, object]] = []
    for item in raw_cues:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        speaker = _normalize_speaker_label(item.get("speaker"))
        body = str(item.get("text") or "").strip()
        if end < start or not body:
            continue
        row = dict(item)
        row["start"] = start
        row["end"] = end
        row["speaker"] = speaker
        row["text"] = body
        normalized = {"start": start, "end": end, "speaker": speaker, "body": body}
        if rows:
            prev_normalized = {
                "start": float(rows[-1].get("start", 0.0) or 0.0),
                "end": float(rows[-1].get("end", 0.0) or 0.0),
                "speaker": str(rows[-1].get("speaker") or ""),
                "body": str(rows[-1].get("text") or ""),
            }
            if _should_merge_export_cue(prev_normalized, normalized, max_gap_seconds=max_gap_seconds):
                rows[-1]["end"] = float(max(float(rows[-1].get("end", end) or end), end))
                rows[-1]["text"] = _join_export_cue_text(str(rows[-1].get("text") or ""), body)
                continue
        rows.append(row)
    out = dict(payload)
    out["cues"] = rows
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["cue_count"] = int(len(rows))
        out["meta"] = meta
    return out


def _normalize_exported_speaker_labels(
    exports: dict[str, str],
    *,
    profile_path: str | Path,
    max_speakers: int,
    profile_remap: dict[str, str] | None = None,
) -> dict[str, object]:
    merged_profile_remap = {
        _normalize_speaker_label(key): _normalize_speaker_label(value)
        for (key, value) in (profile_remap or {}).items()
        if _normalize_speaker_label(key) and _normalize_speaker_label(value)
    }
    duration_remap = _speaker_profile_duration_remap(profile_path, max_speakers=max_speakers)
    merged_profile_remap.update(duration_remap)
    changed_files: list[str] = []
    for path_str in exports.values():
        path = Path(str(path_str or ""))
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".srt"}:
                rewritten = _rewrite_speaker_labels_text(path.read_text(encoding="utf-8"), profile_remap=merged_profile_remap)
                if suffix == ".txt":
                    rewritten = _coalesce_txt_same_speaker_cues(rewritten)
                elif suffix == ".srt":
                    rewritten = _coalesce_srt_same_speaker_cues(rewritten)
                path.write_text(rewritten + ("\n" if rewritten else ""), encoding="utf-8")
                changed_files.append(str(path))
            elif suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                rewritten_payload = _rewrite_json_speaker_labels(payload, profile_remap=merged_profile_remap)
                rewritten_payload = _coalesce_json_same_speaker_cues(rewritten_payload)
                path.write_text(json.dumps(rewritten_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                changed_files.append(str(path))
        except Exception:
            continue
    return {
        "max_speakers": int(max(0, max_speakers)),
        "profile_remap_count": int(len({k: v for (k, v) in merged_profile_remap.items() if k != v})),
        "changed_files": changed_files,
    }


def _should_compare_speakers(profile: str) -> bool:
    return str(profile or "").strip().lower() == "accurate"


def _speaker_compare_disabled_summary(profile: str) -> dict[str, object]:
    return {
        "enabled": False,
        "disabled_reason": f"profile={str(profile or '').strip().lower() or 'unknown'}",
        "reference_sequence": [],
        "realtime_sequence": [],
        "reference_speaker_count": 0,
        "realtime_speaker_count": 0,
        "reference_switch_count": 0,
        "realtime_switch_count": 0,
        "speaker_sequence_distance": 0,
        "speaker_sequence_error_rate": 0.0,
        "realtime_extra_speaker_labels": [],
        "realtime_missing_speaker_labels": [],
    }


def _speaker_compare_summary(reference_text: str, candidate_text: str) -> dict[str, object]:
    reference = _speaker_marker_sequence(reference_text)
    candidate = _speaker_marker_sequence(candidate_text)
    distance = _levenshtein(reference, candidate)
    return {
        "enabled": True,
        "disabled_reason": "",
        "reference_sequence": reference,
        "realtime_sequence": candidate,
        "reference_speaker_count": len(set(reference)),
        "realtime_speaker_count": len(set(candidate)),
        "reference_switch_count": max(0, len(reference) - 1),
        "realtime_switch_count": max(0, len(candidate) - 1),
        "speaker_sequence_distance": int(distance),
        "speaker_sequence_error_rate": float(distance) / float(max(1, len(reference))),
        "realtime_extra_speaker_labels": sorted(set(candidate) - set(reference)),
        "realtime_missing_speaker_labels": sorted(set(reference) - set(candidate)),
    }


def _speaker_count_from_compare_text(text: str) -> int:
    return int(len(set(_speaker_marker_sequence(text))))


def _resolve_realtime_speaker_label_cap(
    *,
    requested_max_speakers: int,
    direct_text_for_compare: str,
    profile: str,
) -> int:
    requested = int(max(0, requested_max_speakers))
    if requested > 0:
        return requested
    if not _should_compare_speakers(profile):
        return 0
    # Realtime uses short overlapping windows, so its raw profile count is much noisier
    # than the long-window WhisperX reference. For compare/export, keep realtime at the
    # same speaker granularity as the reference unless the caller provides an explicit cap.
    return int(max(0, _speaker_count_from_compare_text(direct_text_for_compare)))


def _render_speaker_sequence_html(speaker_compare: dict[str, object]) -> str:
    if not bool(speaker_compare.get("enabled", False)):
        reason = html_lib.escape(str(speaker_compare.get("disabled_reason", "disabled")))
        return f"<div class='speaker-note'>speaker compare disabled ({reason})</div>"
    reference = [str(item) for item in speaker_compare.get("reference_sequence", [])]
    realtime = [str(item) for item in speaker_compare.get("realtime_sequence", [])]

    def _token(label: str, class_name: str) -> str:
        return f"<span class='speaker-token {class_name}'>{html_lib.escape(label)}</span>"

    ref_parts: list[str] = []
    rt_parts: list[str] = []
    matcher = difflib.SequenceMatcher(a=reference, b=realtime)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            ref_parts.extend(_token(label, "same-speaker") for label in reference[i1:i2])
            rt_parts.extend(_token(label, "same-speaker") for label in realtime[j1:j2])
            continue
        if op in {"delete", "replace"}:
            ref_parts.extend(_token(label, "missing") for label in reference[i1:i2])
        if op in {"insert", "replace"}:
            rt_parts.extend(_token(label, "extra") for label in realtime[j1:j2])
    stats = (
        f"reference_speakers={speaker_compare.get('reference_speaker_count', 0)} | "
        f"realtime_speakers={speaker_compare.get('realtime_speaker_count', 0)} | "
        f"sequence_error={float(speaker_compare.get('speaker_sequence_error_rate', 0.0)):.6f}"
    )
    return "\n".join(
        [
            f"<div class='speaker-stats'>{html_lib.escape(stats)}</div>",
            "<div class='speaker-row'><span class='speaker-label'>WhisperX</span> "
            + " ".join(ref_parts)
            + "</div>",
            "<div class='speaker-row'><span class='speaker-label'>Realtime</span> "
            + " ".join(rt_parts)
            + "</div>",
        ]
    )


# Keep intra-word apostrophes attached (straight ' and curly ’) so contractions stay one
# token ("we're", not "we ' re"): both the word-unit metric and the HTML diff render them
# naturally. The char-unit tokenizer already treats ' as a word char (see _tokenize_for_compare).
_WORD_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
_CJK_OR_JP_CHAR_RE = re.compile(r"[\u3400-\u9FFF\u3040-\u30FF]")


def _contains_cjk_or_japanese(text: str) -> bool:
    return bool(_CJK_OR_JP_CHAR_RE.search(str(text or "")))


def _is_cjk_or_japanese_language(language_hint: str | None) -> bool:
    token = str(language_hint or "").strip().lower()
    if not token:
        return False
    return token.startswith("zh") or token.startswith("ja")


def _resolve_compare_unit(reference: str, candidate: str, language_hint: str | None) -> str:
    if _is_cjk_or_japanese_language(language_hint):
        return "char"
    if _contains_cjk_or_japanese(reference) or _contains_cjk_or_japanese(candidate):
        return "char"
    return "word"


def _tokenize_for_compare(text: str, *, unit: str, preserve_newlines: bool = False) -> list[str]:
    src = str(text or "")
    if unit == "char":
        if preserve_newlines:
            normalized = src.replace("\r\n", "\n").replace("\r", "\n")
            normalized = re.sub(r"[ \t]+", " ", normalized)
            normalized = re.sub(r" *\n+ *", "\n", normalized).strip()
        else:
            normalized = re.sub(r"\s+", " ", src).strip()
        if not normalized:
            return []
        tokens: list[str] = []
        i = 0
        n = len(normalized)
        while i < n:
            ch = normalized[i]
            if preserve_newlines and ch == "\n":
                tokens.append("\n")
                i += 1
                continue
            if ch == " ":
                tokens.append(" ")
                i += 1
                continue
            if _CJK_OR_JP_CHAR_RE.match(ch):
                tokens.append(ch)
                i += 1
                continue
            if ch.isascii() and (ch.isalnum() or ch in {"_", "-", "'"}):
                j = i + 1
                while j < n:
                    c = normalized[j]
                    if c.isascii() and (c.isalnum() or c in {"_", "-", "'"}):
                        j += 1
                        continue
                    break
                tokens.append(normalized[i:j])
                i = j
                continue
            tokens.append(ch)
            i += 1
        return tokens
    if not preserve_newlines:
        return _WORD_TOKEN_RE.findall(src)
    tokens: list[str] = []
    lines = src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        if index > 0:
            tokens.append("\n")
        tokens.extend(_WORD_TOKEN_RE.findall(line))
    while tokens and tokens[0] == "\n":
        tokens.pop(0)
    while tokens and tokens[-1] == "\n":
        tokens.pop()
    return tokens


def _join_tokens(tokens: list[str], *, unit: str) -> str:
    if not tokens:
        return ""
    if unit == "char":
        return "".join(tokens)
    out: list[str] = []
    for tok in tokens:
        token = str(tok or "")
        if not token:
            continue
        if token == "\n":
            if out and not out[-1].endswith("\n"):
                out.append("\n")
            continue
        if not out:
            out.append(token)
            continue
        if re.fullmatch(r"[\.,!?;:%\)\]\}]", token):
            out[-1] = out[-1] + token
            continue
        if re.fullmatch(r"[\(\[\{]", out[-1]):
            out[-1] = out[-1] + token
            continue
        out.append(" " + token)
    return "".join(out)


def _word_diff_separator(prev_token: str | None, token: str) -> str:
    """Whitespace to insert before `token` in a word-unit diff, mirroring `_join_tokens`.

    The word tokenizer drops whitespace, so the metric/txt path reinserts it via
    `_join_tokens`; the HTML diff must do the same or every word collapses together.
    """
    if prev_token is None:
        return ""
    if token == "\n" or prev_token == "\n":
        return ""
    if re.fullmatch(r"[\.,!?;:%\)\]\}]", token):
        return ""
    if re.fullmatch(r"[\(\[\{]", prev_token):
        return ""
    return " "


def _render_html_diff(reference_tokens: list[str], candidate_tokens: list[str], *, unit: str) -> str:
    def _render_token_html(token: str) -> str:
        src = str(token or "")
        if src == "\n":
            return "<br>\n"
        if src == " ":
            return "&nbsp;"
        escaped = html_lib.escape(src)
        if src.strip() == "":
            return escaped.replace(" ", "&nbsp;").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        return escaped

    # Flatten the opcodes into an ordered (class, token) stream so word-unit
    # separators can be reinserted between adjacent tokens (the tokenizer drops
    # whitespace, and char-unit keeps spaces as their own tokens already).
    seq: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(a=reference_tokens, b=candidate_tokens)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            seq.extend(("same", tok) for tok in candidate_tokens[j1:j2])
            continue
        if op in {"insert", "replace"}:
            seq.extend(("extra", tok) for tok in candidate_tokens[j1:j2])
        if op in {"delete", "replace"}:
            seq.extend(("missing", tok) for tok in reference_tokens[i1:i2])

    parts: list[str] = []
    buffer: list[str] = []
    current_class: str | None = None
    prev_token: str | None = None

    def _flush() -> None:
        nonlocal current_class
        if buffer and current_class is not None:
            parts.append(f"<span class='{current_class}'>{''.join(buffer)}</span>")
        buffer.clear()

    for class_name, token in seq:
        sep = "" if unit == "char" else _word_diff_separator(prev_token, token)
        if sep:
            _flush()
            current_class = None
            parts.append(sep)
        if class_name != current_class:
            _flush()
            current_class = class_name
        buffer.append(_render_token_html(token))
        prev_token = token
    _flush()
    return "".join(parts)


def _build_reference_diff(
    reference: str,
    candidate: str,
    *,
    language_hint: str | None,
    preserve_newlines: bool = False,
) -> dict[str, object]:
    compare_unit = _resolve_compare_unit(reference, candidate, language_hint)
    reference_tokens = _tokenize_for_compare(reference, unit=compare_unit, preserve_newlines=preserve_newlines)
    candidate_tokens = _tokenize_for_compare(candidate, unit=compare_unit, preserve_newlines=preserve_newlines)

    matcher = difflib.SequenceMatcher(a=reference_tokens, b=candidate_tokens)
    marker: list[str] = []
    same = 0
    candidate_extra = 0
    candidate_missing = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            span = max(0, i2 - i1)
            same += span
            marker.extend(["."] * span)
            continue
        if op in {"insert", "replace"}:
            span = max(0, j2 - j1)
            candidate_extra += span
            marker.extend(["-"] * span)
        if op in {"delete", "replace"}:
            span = max(0, i2 - i1)
            candidate_missing += span
            marker.extend(["+"] * span)

    reference_aligned = _join_tokens(reference_tokens, unit=compare_unit)
    candidate_aligned = _join_tokens(candidate_tokens, unit=compare_unit)
    html_annotated = _render_html_diff(reference_tokens, candidate_tokens, unit=compare_unit)
    return {
        "compare_unit": compare_unit,
        "reference_aligned": reference_aligned,
        "candidate_aligned": candidate_aligned,
        "marker_line": "".join(marker),
        "same_count": int(same),
        "candidate_extra_count": int(candidate_extra),
        "candidate_missing_count": int(candidate_missing),
        "realtime_annotated_html": html_annotated,
    }


def _render_diff_compare_section(
    *,
    title: str,
    meta_line: str,
    reference_label: str,
    reference_text: str,
    candidate_label: str,
    candidate_annotated_html: str,
    marker_line: str,
    marker_legend: str,
    extra_blocks: list[str] | None = None,
) -> list[str]:
    """Render one compare section (heading + reference / annotated-candidate / marker blocks)."""
    lines = [
        f"  <h2>{html_lib.escape(title)}</h2>",
        f"  <div class='meta'>{meta_line}</div>",
        "  <div class='block'>",
        f"    <div class='label'>{html_lib.escape(reference_label)}</div>",
        html_lib.escape(reference_text).strip(),
        "  </div>",
        "  <div class='block'>",
        f"    <div class='label'>{html_lib.escape(candidate_label)} (extra=deep red strike, missing=deep green insert)</div>",
        str(candidate_annotated_html).strip(),
        "  </div>",
    ]
    for block in (extra_blocks or []):
        lines.append(str(block))
    lines += [
        "  <div class='block mono'>",
        "    <div class='label'>Marker</div>",
        f"    {html_lib.escape(marker_legend)}",
        f"    \n{html_lib.escape(marker_line)}",
        "  </div>",
    ]
    return lines


def _build_compare_html_page(
    *,
    input_path: str,
    compare_unit: str,
    cer: float,
    distance: int,
    reference_text: str,
    realtime_annotated_html: str,
    marker_line: str,
    speaker_compare_html: str = "",
    reference_section_html: str = "",
) -> str:
    body: list[str] = [
        "<body>",
        "  <h1>Compare Report</h1>",
        f"  <div class='meta'>input={html_lib.escape(input_path)}</div>",
    ]
    if str(reference_section_html).strip():
        body.append(str(reference_section_html))
        body.append("  <hr>")
    body += _render_diff_compare_section(
        title="WhisperX vs Realtime",
        meta_line=(
            f"unit={html_lib.escape(compare_unit)} | normalized_distance={distance} | "
            f"normalized_ratio={cer:.6f}"
        ),
        reference_label="WhisperX Reference",
        reference_text=reference_text,
        candidate_label="Realtime",
        candidate_annotated_html=realtime_annotated_html,
        marker_line=marker_line,
        marker_legend=". = same | - = realtime extra | + = realtime missing",
        extra_blocks=[
            "  <div class='block'>",
            "    <div class='label'>Speaker Compare</div>",
            str(speaker_compare_html).strip(),
            "  </div>",
        ],
    )
    body.append("</body>")
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1'>",
            "  <title>Compare Report</title>",
            "  <style>",
            "    body { font-family: Segoe UI, Noto Sans CJK TC, Arial, sans-serif; margin: 20px; color: #111; }",
            "    h1 { margin: 0 0 8px 0; font-size: 24px; }",
            "    h2 { margin: 18px 0 8px 0; font-size: 19px; }",
            "    hr { border: none; border-top: 2px solid #ccc; margin: 22px 0; }",
            "    .meta { margin: 0 0 14px 0; color: #444; font-size: 13px; }",
            "    .block { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 10px 0; white-space: pre-wrap; line-height: 1.7; font-weight: 400; }",
            "    .label { font-weight: 700; margin-bottom: 6px; }",
            "    .same { color: #111; font-weight: 400; text-decoration: none; }",
            "    .extra { color: #8B0000; font-weight: 400; text-decoration: line-through; text-decoration-thickness: 2px; }",
            "    .missing { color: #0B5D1E; font-weight: 700; }",
            "    .mono { font-family: Consolas, Menlo, monospace; white-space: pre-wrap; }",
            "    .speaker-row { margin: 6px 0; line-height: 1.9; }",
            "    .speaker-label { display: inline-block; min-width: 80px; font-weight: 700; }",
            "    .speaker-token { border-radius: 4px; padding: 2px 5px; margin: 0 2px; font-family: Consolas, Menlo, monospace; }",
            "    .same-speaker { color: #111; background: #f2f2f2; font-weight: 400; }",
            "    .speaker-stats, .speaker-note { color: #444; font-size: 13px; margin-bottom: 8px; }",
            "  </style>",
            "</head>",
            *body,
            "</html>",
        ]
    )


# Canonical source ordering: lower index = treated as the diff "reference" (a) so that
# extra/missing read as the candidate (b) over/under-producing vs the more ground-truth side.
_COMPARE_SOURCE_ORDER = ("srt", "spk_subtitles", "direct", "realtime")
_COMPARE_SOURCE_LABELS = {
    "srt": "SRT 參考字幕",
    "spk_subtitles": "spk_subtitles (含說話人 GT)",
    "direct": "Direct (整檔)",
    "realtime": "Realtime (即時)",
}


def _build_interactive_compare_html(
    *,
    input_path: str,
    meta_line: str,
    sources: dict[str, dict[str, object]],
    language_hint: str | None,
    spk_vs_truth: dict | None,
) -> str:
    """Interactive compare page: pick any 2 of {srt, spk_subtitles, direct, realtime},
    toggle 有 spk / 沒 spk, see their (S/T-folded) diff + a metrics table.

    `sources[key]` = {"available": bool, "nospk": str|None, "spk": str|None}; a source is
    selectable only where its text for the chosen mode is non-empty. All diffs are pre-rendered
    server-side (reusing the S/T-folding `_normalize_for_html_compare` + `_build_reference_diff`)
    and toggled client-side, so no diff/opencc logic is reimplemented in JS.
    """
    order = [k for k in _COMPARE_SOURCE_ORDER if k in sources]
    avail: dict[str, dict[str, bool]] = {}
    for k in order:
        avail[k] = {
            "nospk": bool(str(sources[k].get("nospk") or "").strip()),
            "spk": bool(str(sources[k].get("spk") or "").strip()),
        }

    def _label(k: str) -> str:
        return _COMPARE_SOURCE_LABELS.get(k, k)

    blocks: list[str] = []
    rows: list[dict[str, object]] = []
    for mode in ("nospk", "spk"):
        elig = [k for k in order if avail[k][mode]]
        for ai in range(len(elig)):
            for bi in range(ai + 1, len(elig)):
                a, b = elig[ai], elig[bi]
                ta = _normalize_for_html_compare(str(sources[a].get(mode) or ""))
                tb = _normalize_for_html_compare(str(sources[b].get(mode) or ""))
                d = _build_reference_diff(ta, tb, language_hint=language_hint, preserve_newlines=True)
                unit = str(d.get("compare_unit", "char"))
                same = int(d.get("same_count", 0))
                extra = int(d.get("candidate_extra_count", 0))
                missing = int(d.get("candidate_missing_count", 0))
                ref_len = same + missing
                ratio = (float(extra + missing) / float(ref_len)) if ref_len else 0.0
                bid = f"diff__{a}__{b}__{mode}"
                spk_acc: float | None = None
                if mode == "spk" and a == "spk_subtitles" and isinstance(spk_vs_truth, dict):
                    if b == "direct":
                        spk_acc = spk_vs_truth.get("direct_speaker_accuracy")
                    elif b == "realtime":
                        spk_acc = spk_vs_truth.get("realtime_speaker_accuracy")
                inner = "\n".join(
                    _render_diff_compare_section(
                        title=f"{_label(a)}  →  {_label(b)}",
                        meta_line=(
                            f"mode={'有 spk' if mode == 'spk' else '沒 spk'} | unit={html_lib.escape(unit)} | "
                            f"same={same} | {html_lib.escape(_label(b))}-extra={extra} | "
                            f"{html_lib.escape(_label(b))}-missing={missing} | ratio={ratio:.4f}"
                        ),
                        reference_label=f"{_label(a)} (reference)",
                        reference_text=str(d.get("reference_aligned", "")),
                        candidate_label=_label(b),
                        candidate_annotated_html=str(d.get("realtime_annotated_html", "")),
                        marker_line=str(d.get("marker_line", "")),
                        marker_legend=(
                            f". = same | - = {_label(b)} extra | + = {_label(b)} missing (即 {_label(a)} 有而 {_label(b)} 無)"
                        ),
                    )
                )
                blocks.append(f"  <div class='diffblock' id='{bid}' style='display:none'>\n{inner}\n  </div>")
                rows.append({
                    "a": a, "b": b, "mode": mode, "unit": unit, "same": same,
                    "extra": extra, "missing": missing, "ratio": ratio,
                    "spk_acc": spk_acc, "bid": bid,
                })

    # ---- controls (checkboxes + mode radios) ----
    checkbox_html: list[str] = []
    for k in order:
        any_avail = avail[k]["nospk"] or avail[k]["spk"]
        disabled = "" if any_avail else " disabled"
        note = "" if any_avail else " <span class='muted'>(無此來源)</span>"
        spk_only_note = "" if avail[k]["spk"] else " <span class='muted'>(無說話人標註)</span>"
        checkbox_html.append(
            f"<label class='srcbox{'' if any_avail else ' off'}'>"
            f"<input type='checkbox' class='srcchk' value='{k}'{disabled}> {html_lib.escape(_label(k))}"
            f"{note}{spk_only_note}</label>"
        )

    # ---- metrics table ----
    table_rows: list[str] = []
    for r in rows:
        acc = "" if r["spk_acc"] is None else f"{float(r['spk_acc']) * 100:.1f}%"
        table_rows.append(
            f"<tr id='row__{r['bid']}'>"
            f"<td>{html_lib.escape(_label(str(r['a'])))} → {html_lib.escape(_label(str(r['b'])))}</td>"
            f"<td>{'有 spk' if r['mode'] == 'spk' else '沒 spk'}</td>"
            f"<td>{html_lib.escape(str(r['unit']))}</td>"
            f"<td class='num'>{r['same']}</td>"
            f"<td class='num'>{r['extra']}</td>"
            f"<td class='num'>{r['missing']}</td>"
            f"<td class='num'>{float(r['ratio']):.4f}</td>"
            f"<td class='num'>{acc}</td>"
            f"</tr>"
        )
    table_html = (
        "<table class='metrics'><thead><tr>"
        "<th>來源對 (reference → candidate)</th><th>模式</th><th>unit</th>"
        "<th>same</th><th>extra</th><th>missing</th><th>ratio</th><th>spk_acc</th>"
        "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
    )

    avail_js = json.dumps(avail, ensure_ascii=False)
    order_js = json.dumps(order, ensure_ascii=False)
    script = (
        "<script>\n"
        "const ORDER = " + order_js + ";\n"
        "const AVAIL = " + avail_js + ";\n"
        "let checkedOrder = [];\n"
        "function modeVal(){ const m = document.querySelector('input[name=mode]:checked'); return m ? m.value : 'nospk'; }\n"
        "function render(){\n"
        "  document.querySelectorAll('.diffblock').forEach(e => e.style.display='none');\n"
        "  document.querySelectorAll('table.metrics tr').forEach(e => e.classList.remove('active'));\n"
        "  const hint = document.getElementById('hint');\n"
        "  const mode = modeVal();\n"
        "  if (checkedOrder.length !== 2){ hint.textContent = '請勾選兩個來源 (目前 ' + checkedOrder.length + ' 個)'; hint.style.display='block'; return; }\n"
        "  let pair = checkedOrder.slice().sort((x,y)=>ORDER.indexOf(x)-ORDER.indexOf(y));\n"
        "  const a = pair[0], b = pair[1];\n"
        "  const bid = 'diff__' + a + '__' + b + '__' + mode;\n"
        "  const block = document.getElementById(bid);\n"
        "  if (!block){\n"
        "    const bad = (!AVAIL[a][mode] ? a : (!AVAIL[b][mode] ? b : null));\n"
        "    hint.textContent = bad ? ('「' + bad + '」在此模式下無資料 (例如 srt 無說話人標註)，請切換模式或改選來源。') : '此組合無資料。';\n"
        "    hint.style.display='block'; return;\n"
        "  }\n"
        "  hint.style.display='none';\n"
        "  block.style.display='block';\n"
        "  const row = document.getElementById('row__' + bid);\n"
        "  if (row) row.classList.add('active');\n"
        "}\n"
        "document.querySelectorAll('.srcchk').forEach(chk => chk.addEventListener('change', () => {\n"
        "  if (chk.checked){ checkedOrder.push(chk.value); }\n"
        "  else { checkedOrder = checkedOrder.filter(v => v !== chk.value); }\n"
        "  while (checkedOrder.length > 2){\n"
        "    const drop = checkedOrder.shift();\n"
        "    const el = document.querySelector('.srcchk[value=\"' + drop + '\"]');\n"
        "    if (el) el.checked = false;\n"
        "  }\n"
        "  render();\n"
        "}));\n"
        "document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', render));\n"
        "render();\n"
        "</script>\n"
    )

    body = [
        "<body>",
        "  <h1>Compare Report (互動式)</h1>",
        f"  <div class='meta'>input={html_lib.escape(input_path)}</div>",
        f"  <div class='meta'>{meta_line}</div>",
        "  <div class='controls'>",
        "    <div class='row'><span class='ctl-label'>來源 (勾選兩個):</span> " + " ".join(checkbox_html) + "</div>",
        "    <div class='row'><span class='ctl-label'>模式:</span> "
        "<label><input type='radio' name='mode' value='nospk' checked> 沒 spk (純文字)</label> "
        "<label><input type='radio' name='mode' value='spk'> 有 spk (含說話人)</label></div>",
        "  </div>",
        "  <div id='hint' class='hint'></div>",
        "  <h2>數據表</h2>",
        "  " + table_html,
        "  <h2>Diff</h2>",
        "\n".join(blocks),
        script,
        "</body>",
    ]
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='zh-Hant'>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1'>",
            "  <title>Compare Report</title>",
            "  <style>",
            "    body { font-family: Segoe UI, Noto Sans CJK TC, Arial, sans-serif; margin: 20px; color: #111; }",
            "    h1 { margin: 0 0 8px 0; font-size: 24px; }",
            "    h2 { margin: 18px 0 8px 0; font-size: 19px; }",
            "    hr { border: none; border-top: 2px solid #ccc; margin: 22px 0; }",
            "    .meta { margin: 0 0 8px 0; color: #444; font-size: 13px; }",
            "    .controls { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 12px 0; background: #fafafa; }",
            "    .controls .row { margin: 6px 0; line-height: 2.0; }",
            "    .ctl-label { font-weight: 700; margin-right: 8px; }",
            "    .srcbox { display: inline-block; margin-right: 14px; padding: 2px 6px; border-radius: 6px; }",
            "    .srcbox.off { color: #999; }",
            "    .muted { color: #999; font-size: 12px; }",
            "    .hint { display:none; color: #8B5A00; background:#FFF7E6; border:1px solid #FFE0A3; border-radius:6px; padding:8px 12px; margin:10px 0; }",
            "    table.metrics { border-collapse: collapse; margin: 8px 0 4px 0; font-size: 13px; }",
            "    table.metrics th, table.metrics td { border: 1px solid #ddd; padding: 4px 10px; text-align: left; }",
            "    table.metrics th { background: #f2f2f2; }",
            "    table.metrics td.num { text-align: right; font-family: Consolas, Menlo, monospace; }",
            "    table.metrics tr.active { background: #FFF4CC; font-weight: 700; }",
            "    .block { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 10px 0; white-space: pre-wrap; line-height: 1.7; font-weight: 400; }",
            "    .label { font-weight: 700; margin-bottom: 6px; }",
            "    .same { color: #111; font-weight: 400; text-decoration: none; }",
            "    .extra { color: #8B0000; font-weight: 400; text-decoration: line-through; text-decoration-thickness: 2px; }",
            "    .missing { color: #0B5D1E; font-weight: 700; }",
            "    .mono { font-family: Consolas, Menlo, monospace; white-space: pre-wrap; }",
            "  </style>",
            "</head>",
            *body,
            "</html>",
        ]
    )


def _join_words_for_compare(words: list[str]) -> str:
    out: list[str] = []
    prev = ""
    for token in words:
        w = str(token or "").strip()
        if not w:
            continue
        if not out:
            out.append(w)
            prev = w
            continue
        if re.fullmatch(r"[\.,!?;:\)\]\}，。！？；：、）】」』]+", w):
            out[-1] = out[-1] + w
            prev = w
            continue
        if re.search(r"[\u3400-\u9FFF]", w) or re.search(r"[\u3400-\u9FFF]", prev):
            out.append(w)
        else:
            out.append(" " + w)
        prev = w
    return "".join(out).strip()


def _build_direct_grouped_text(meta: dict[str, object], *, group_seconds: float) -> str:
    rows = meta.get("token_timestamps")
    if not isinstance(rows, list) or group_seconds <= 0.0:
        return ""
    groups: dict[int, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        word = str(row.get("word") or "").strip()
        if not word:
            continue
        try:
            start = float(row.get("absolute_start", row.get("start")))
        except Exception:
            continue
        if start < 0.0:
            continue
        gid = int(start // group_seconds)
        groups.setdefault(gid, []).append(word)
    if not groups:
        return ""
    lines: list[str] = []
    for gid in sorted(groups.keys()):
        start = float(gid) * float(group_seconds)
        end = start + float(group_seconds)
        content = _join_words_for_compare(groups.get(gid, []))
        if not content:
            continue
        lines.append(f"[{start:07.2f}-{end:07.2f}] {content}")
    return "\n".join(lines).strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _make_status_callback(prefix: str):
    def _cb(message: str) -> None:
        msg = str(message or "").strip()
        if not msg:
            return
        print(f"[{prefix}] {msg}", flush=True)

    return _cb


def _resolve_transcriber_device(transcriber: object) -> str:
    """Lowercase device string for a transcriber (e.g. 'cuda', 'vulkan', 'cpu').

    Plain WhisperX/whisper.cpp-subprocess transcribers expose `_device` directly.
    whisper.cpp's default resident-server mode (WhisperCppServerTranscriber,
    stt/whispercpp_server.py) does not set `_device` on itself -- the device lives
    on its inner `_manager.device` instead -- so fall back to that before giving up.
    """
    device = str(getattr(transcriber, "_device", "") or "").strip()
    if not device:
        manager = getattr(transcriber, "_manager", None)
        device = str(getattr(manager, "device", "") or "").strip()
    return device.lower()


def _describe_transcriber(transcriber: object) -> str:
    asr_device = _resolve_transcriber_device(transcriber) or "unknown"
    align_device = str(getattr(transcriber, "_align_device", "n/a"))
    compute_type = str(getattr(transcriber, "_compute_type", "unknown"))
    return f"asr_device={asr_device}; align_device={align_device}; compute_type={compute_type}"


def _warmup_transcriber_like_main(transcriber: object, cfg: RuntimeConfig, *, prefix: str) -> None:
    """Mirror AppController warmup so compare realtime has the same model-cache state."""
    if transcriber is None:
        return
    warmup_scope = "VAD/cache pre-init"
    if bool(getattr(cfg, "whisperx_enable_diarization", False)):
        warmup_scope = "VAD/cache/diarization pre-init"
    print(f"[{prefix}] WhisperX warmup started ({warmup_scope}).", flush=True)
    started = time.monotonic()
    try:
        prewarm_fn = getattr(transcriber, "prewarm", None)
        if callable(prewarm_fn):
            prewarm_fn(getattr(cfg, "source_language", None))
        sample_rate = 16000
        channels = 1
        pcm = b"\x00\x00" * int(sample_rate * channels)
        warmup_chunk = AudioChunk(pcm16=pcm, sample_rate=sample_rate, channels=channels)
        transcribe_fn = getattr(transcriber, "transcribe", None)
        if callable(transcribe_fn):
            transcribe_fn(
                warmup_chunk,
                language=getattr(cfg, "source_language", None),
                channel_mode=str(getattr(cfg, "source_channel_mode", "mono") or "mono"),
            )
        print(
            f"[{prefix}] WhisperX warmup completed in {time.monotonic() - started:.2f}s.",
            flush=True,
        )
    except Exception as exc:
        print(f"[{prefix}] WhisperX warmup failed: {exc}", flush=True)


def _load_base_cfg() -> RuntimeConfig:
    cfg = RuntimeConfig()
    updates = load_persisted_updates()
    apply_updates_to_config(cfg, updates)
    cfg.stt_provider = "whisperx"
    cfg.stt_variant = "gpu"
    cfg.model_device = "cuda"
    cfg.compute_type = "float16"
    return cfg


def _resolve_model_ref(model_arg: str, model_root: Path, persisted_model: str) -> str:
    token = str(model_arg or "").strip()
    if not token:
        token = str(persisted_model or "").strip()
    if not token:
        token = "medium"

    candidate = Path(token)
    if candidate.exists():
        return str(candidate.resolve())

    if not candidate.is_absolute():
        under_root = (model_root / token)
        if under_root.exists():
            return str(under_root.resolve())
    return token


def _parse_formats(raw: str) -> list[str]:
    formats: list[str] = []
    for item in str(raw or "").split(","):
        token = item.strip().lower()
        if token in {"txt", "srt", "json"} and token not in formats:
            formats.append(token)
    return formats or ["txt", "srt", "json"]


def _try_read_text(path_str: str) -> str:
    try:
        path = Path(str(path_str or "")).resolve()
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore").lstrip(chr(0xFEFF))
    except Exception:
        return ""
    return ""


def _create_export_session(
    *,
    out_dir: Path,
    formats: list[str],
    include_timestamps: bool,
    include_speaker: bool,
) -> TranscriptExporterSession:
    opts = TranscriptExportOptions(
        enabled=True,
        formats=formats,
        include_timestamps=include_timestamps,
        include_speaker=include_speaker,
        output_dir=str(out_dir),
    )
    return TranscriptExporterSession(opts)


def _prepare_case_speaker_profile_path(case_dir: Path, name: str) -> str:
    profile_dir = case_dir / "_speaker_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{name}_profiles.json"
    for target in (path, path.with_suffix(path.suffix + ".tmp")):
        try:
            target.unlink(missing_ok=True)
        except Exception:
            # Use a fresh filename if a previous run is still holding the file.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = profile_dir / f"{name}_{stamp}_profiles.json"
            break
    return str(path)


def _finalize_and_rename_exports(
    session: TranscriptExporterSession,
    *,
    target_dir: Path,
    prefix: str,
) -> dict[str, str]:
    written = session.finalize()
    out: dict[str, str] = {}
    for path in written:
        suffix = path.suffix.lower()
        target = target_dir / f"{prefix}{suffix}"
        try:
            path.replace(target)
        except Exception:
            shutil.copy2(path, target)
        out[suffix.lstrip(".")] = str(target)
    return out


def _write_realtime_main_payload_exports(
    *,
    target_dir: Path,
    prefix: str,
    formats: list[str],
    text: str,
    duration_seconds: float,
) -> dict[str, str]:
    """Write realtime artifacts from the exact main overlay payload.

    The compare realtime phase is intended to reflect what `main` would show on
    screen. Do not pass this text through transcript exporter snapshot collapse,
    speaker remapping, or same-speaker cue coalescing.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = str(text or "").strip()
    out: dict[str, str] = {}
    for fmt in formats:
        suffix = str(fmt or "").strip().lower()
        if suffix == "txt":
            path = target_dir / f"{prefix}.txt"
            path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
            out["txt"] = str(path)
        elif suffix == "srt":
            path = target_dir / f"{prefix}.srt"
            if payload:
                end = max(0.001, float(duration_seconds or 0.0))
                content = "\n".join(
                    [
                        "1",
                        f"00:00:00,000 --> {_format_export_time_srt(end)}",
                        payload,
                        "",
                    ]
                )
            else:
                content = ""
            path.write_text(content, encoding="utf-8")
            out["srt"] = str(path)
        elif suffix == "json":
            path = target_dir / f"{prefix}.json"
            data = {
                "meta": {
                    "source": "main_overlay_payload",
                    "duration_seconds": float(max(0.0, duration_seconds or 0.0)),
                    "cue_count": 1 if payload else 0,
                },
                "cues": [
                    {
                        "start": 0.0,
                        "end": float(max(0.001, duration_seconds or 0.0)),
                        "speaker": "",
                        "text": payload,
                    }
                ]
                if payload
                else [],
                "text": payload,
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            out["json"] = str(path)
    return out


def _reconcile_incremental_speaker_profiles(transcriber: object, *, threshold: float = 0.0) -> dict[str, object]:
    reconcile = getattr(transcriber, "reconcile_speaker_profiles", None)
    try:
        stats = _call_speaker_profile_reconcile(reconcile, threshold=threshold)
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "merged_count": 0, "remap": {}}
    return dict(stats)


def _speaker_reconcile_remap(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("remap")
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for (key, value) in raw.items()}


def _build_runtime_cfg(
    base: RuntimeConfig,
    *,
    model_ref: str,
    language: str | None,
    segment_seconds: float,
    hop_seconds: float,
    device: str,
    profile: str,
    align_model: str,
    align_language: str,
    align_device: str,
    force_alignment: str,
    compute_type: str,
    beam_size: int,
    batch_size: int,
    speaker_profile_match_threshold: float,
    speaker_profile_min_seconds: float,
    speaker_profile_reconcile_threshold: float,
    speaker_realtime_refresh_seconds: float | None = None,
    speaker_realtime_refresh_alpha: float | None = None,
    speaker_realtime_refresh_assign_threshold: float | None = None,
    speaker_realtime_refresh_min_cluster_seconds: float | None = None,
    speaker_realtime_refresh_merge: bool | None = None,
    speaker_count_hint_enabled: bool | None = None,
    speaker_count_hint_seconds: float | None = None,
    speaker_count_hint_window_seconds: float | None = None,
    speaker_count_hint_sliver_floor_seconds: float | None = None,
    speaker_merge_grace_windows: int | None = None,
    speaker_merge_grace_relief: float | None = None,
    speaker_merge_preserve_centroid: bool | None = None,
    speaker_profile_max_exemplars: int | None = None,
    speaker_profile_exemplar_diversity_threshold: float | None = None,
    subtitle_relabel_enabled: bool | None = None,
    subtitle_relabel_window_seconds: float | None = None,
    subtitle_relabel_sliver_floor_seconds: float | None = None,
    subtitle_relabel_assign_threshold: float | None = None,
    subtitle_relabel_margin: float | None = None,
    subtitle_relabel_async: bool | None = None,
    asr_temperatures: str | None = None,
    asr_log_prob_threshold: float | None = None,
    asr_compression_ratio_threshold: float | None = None,
    asr_no_speech_threshold: float | None = None,
    diarization_min_speakers: int | None = None,
    diarization_max_speakers: int | None = None,
    speaker_profile_quality_gate: bool = False,
) -> RuntimeConfig:
    cfg = RuntimeConfig(**base.__dict__)
    cfg.whisperx_speaker_profile_quality_gate_enabled = bool(speaker_profile_quality_gate)
    cfg.stt_model_path = model_ref
    cfg.segment_seconds = float(segment_seconds)
    cfg.hop_seconds = float(hop_seconds)
    if device != "auto":
        cfg.model_device = device
    cfg.compute_type = str(compute_type or "float16")
    cfg.whisper_beam_size = max(1, int(beam_size or 5))
    cfg.whisper_batch_size = max(1, int(batch_size or 4))
    if profile == "fast":
        cfg.whisperx_enable_diarization = False
        cfg.whisperx_enable_vad = True
        cfg.whisperx_vad_method = "silero"
        cfg.whisperx_enable_forced_alignment = True
        cfg.whisperx_speaker_profile_enabled = False
    if language is None:
        cfg.source_language = None
    elif language and language != "auto":
        cfg.source_language = language
    align_model_token = str(align_model or "").strip()
    if align_model_token:
        cfg.whisperx_alignment_model = align_model_token
    align_lang_token = str(align_language or "").strip()
    if align_lang_token:
        cfg.whisperx_alignment_language = align_lang_token
    if str(align_device) != "auto":
        cfg.whisperx_alignment_device = str(align_device)
    fa = str(force_alignment or "auto").strip().lower()
    if fa == "on":
        cfg.whisperx_enable_forced_alignment = True
    elif fa == "off":
        cfg.whisperx_enable_forced_alignment = False
    if speaker_profile_match_threshold > 0.0:
        cfg.whisperx_speaker_profile_match_threshold = float(
            max(0.0, min(0.999, speaker_profile_match_threshold))
        )
    if speaker_profile_min_seconds > 0.0:
        cfg.whisperx_speaker_profile_min_seconds = float(max(0.2, speaker_profile_min_seconds))
    if speaker_profile_reconcile_threshold > 0.0:
        cfg.whisperx_speaker_profile_reconcile_threshold = float(
            max(0.0, min(0.999, speaker_profile_reconcile_threshold))
        )
    if speaker_realtime_refresh_seconds is not None:
        cfg.whisperx_speaker_realtime_refresh_seconds = float(max(0.0, speaker_realtime_refresh_seconds))
    if speaker_realtime_refresh_alpha is not None:
        cfg.whisperx_speaker_realtime_refresh_alpha = float(max(0.0, min(1.0, speaker_realtime_refresh_alpha)))
    if speaker_realtime_refresh_assign_threshold is not None:
        cfg.whisperx_speaker_realtime_refresh_assign_threshold = float(
            max(0.0, min(0.999, speaker_realtime_refresh_assign_threshold))
        )
    if speaker_realtime_refresh_min_cluster_seconds is not None:
        cfg.whisperx_speaker_realtime_refresh_min_cluster_seconds = float(
            max(0.0, speaker_realtime_refresh_min_cluster_seconds)
        )
    if speaker_realtime_refresh_merge is not None:
        cfg.whisperx_speaker_realtime_refresh_merge = bool(speaker_realtime_refresh_merge)
    if speaker_count_hint_enabled is not None:
        cfg.whisperx_speaker_count_hint_enabled = bool(speaker_count_hint_enabled)
    if speaker_count_hint_seconds is not None:
        cfg.whisperx_speaker_count_hint_seconds = float(max(0.1, speaker_count_hint_seconds))
    if speaker_count_hint_window_seconds is not None:
        cfg.whisperx_speaker_count_hint_window_seconds = float(max(1.0, speaker_count_hint_window_seconds))
    if speaker_count_hint_sliver_floor_seconds is not None:
        cfg.whisperx_speaker_count_hint_sliver_floor_seconds = float(max(0.0, speaker_count_hint_sliver_floor_seconds))
    if speaker_merge_grace_windows is not None:
        cfg.whisperx_speaker_merge_grace_windows = int(max(0, speaker_merge_grace_windows))
    if speaker_merge_grace_relief is not None:
        cfg.whisperx_speaker_merge_grace_relief = float(max(0.0, speaker_merge_grace_relief))
    if speaker_merge_preserve_centroid is not None:
        cfg.whisperx_speaker_merge_preserve_centroid = bool(speaker_merge_preserve_centroid)
    if speaker_profile_max_exemplars is not None:
        cfg.whisperx_speaker_profile_max_exemplars = int(max(1, speaker_profile_max_exemplars))
    if speaker_profile_exemplar_diversity_threshold is not None:
        cfg.whisperx_speaker_profile_exemplar_diversity_threshold = float(
            max(0.0, speaker_profile_exemplar_diversity_threshold)
        )
    if subtitle_relabel_enabled is not None:
        cfg.subtitle_relabel_enabled = bool(subtitle_relabel_enabled)
    if subtitle_relabel_window_seconds is not None:
        cfg.subtitle_relabel_window_seconds = float(max(1.0, subtitle_relabel_window_seconds))
    if subtitle_relabel_sliver_floor_seconds is not None:
        cfg.subtitle_relabel_sliver_floor_seconds = float(max(0.0, subtitle_relabel_sliver_floor_seconds))
    if subtitle_relabel_assign_threshold is not None:
        cfg.subtitle_relabel_assign_threshold = float(
            max(0.0, min(0.999, subtitle_relabel_assign_threshold))
        )
    if subtitle_relabel_margin is not None:
        cfg.subtitle_relabel_margin = float(max(0.0, min(1.0, subtitle_relabel_margin)))
    if subtitle_relabel_async is not None:
        cfg.subtitle_relabel_async = bool(subtitle_relabel_async)
    if asr_temperatures is not None:
        cfg.whisperx_asr_temperatures = str(asr_temperatures)
    if asr_log_prob_threshold is not None:
        cfg.whisperx_asr_log_prob_threshold = float(asr_log_prob_threshold)
    if asr_compression_ratio_threshold is not None:
        cfg.whisperx_asr_compression_ratio_threshold = float(asr_compression_ratio_threshold)
    if asr_no_speech_threshold is not None:
        cfg.whisperx_asr_no_speech_threshold = float(asr_no_speech_threshold)
    if diarization_min_speakers is not None:
        cfg.whisperx_diarization_min_speakers = int(max(0, diarization_min_speakers))
    if diarization_max_speakers is not None:
        cfg.whisperx_diarization_max_speakers = int(max(0, diarization_max_speakers))
    return cfg


def _decode_knob_summary(cfg: RuntimeConfig) -> dict[str, object]:
    return {
        "compute_type": str(getattr(cfg, "compute_type", "") or "float16"),
        "beam_size": int(getattr(cfg, "whisper_beam_size", 5) or 5),
        "batch_size": int(getattr(cfg, "whisper_batch_size", 4) or 4),
    }


def _run_incremental(
    cfg: RuntimeConfig,
    full_audio: AudioChunk,
    *,
    exporter: TranscriptExporterSession,
    require_gpu: bool,
    debug_trace_path: Path | None = None,
    speaker_profile_reconcile_threshold: float = 0.0,
    overlay_snapshot_seconds: float = 30.0,
    overlay_timeline_path: Path | None = None,
    replay_speed: float = 0.0,
) -> dict[str, object]:
    del full_audio  # File replay capture is the realtime input; this keeps the public call shape stable.
    transcriber = None
    capture = None
    trace_handle = None
    try:
        cfg.source_mode = "file"
        source_file = str(getattr(cfg, "source_file_path", "") or "").strip()
        if not source_file:
            raise RuntimeError("incremental realtime-project path requires cfg.source_file_path.")
        cfg.source_file_replay_speed = float(replay_speed)
        cfg.source_file_chunk_seconds = max(0.02, float(getattr(cfg, "source_file_chunk_seconds", 0.25) or 0.25))
        if debug_trace_path is not None:
            cfg.debug_mode = True
            debug_trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = debug_trace_path.open("w", encoding="utf-8")

        transcriber = create_stt_transcriber(cfg, progress_callback=_make_status_callback("incremental/status"))
        print(f"[incremental] transcriber: {_describe_transcriber(transcriber)}", flush=True)
        device_str = _resolve_transcriber_device(transcriber)
        # whisper.cpp reports "vulkan" (never "cuda") for its GPU path; accept either GPU
        # marker here. WhisperX (the direct pass, and the realtime default) still expects cuda.
        if require_gpu and device_str not in ("cuda", "vulkan"):
            raise RuntimeError(
                "incremental path is not on a GPU device (cuda/vulkan); "
                f"got device={device_str!r}. Abort by --allow-cpu-fallback to continue."
            )
        _warmup_transcriber_like_main(transcriber, cfg, prefix="incremental")
        assembler = SubtitleAssembler()
        preprocess_pipeline = create_audio_preprocessing_pipeline(cfg)
        capture = build_capture_from_config(
            cfg,
            on_status=lambda message: print(f"[incremental/source] {message}", flush=True),
        )
        capture.start()
        replay_duration_seconds = 0.0
        duration_probe = getattr(capture, "duration_seconds", None)
        if callable(duration_probe):
            try:
                replay_duration_seconds = float(duration_probe())
            except Exception:
                replay_duration_seconds = 0.0
        print(
            "[incremental] replay source: "
            f"path={cfg.source_file_path}; segment={cfg.segment_seconds:g}s; hop={cfg.hop_seconds:g}s",
            flush=True,
        )

        final_text = ""
        accumulated_text = ""
        raw_hits = 0
        recorded_event_count = 0
        debug_event_count = 0
        last_progress_report = float("-inf")
        # Round 0017: capture the live overlay frame (committed | separator | raw)
        # on a fixed audio-time cadence, so the realtime on-screen experience can be
        # eyeballed offline. Kept OUT of final_text/accumulated_text (those drive CER
        # + export and must stay the clean source).
        current_audio_elapsed = 0.0
        overlay_timeline: list[tuple[float, str]] = []
        last_overlay_snapshot_s = float("-inf")
        started = time.monotonic()
        running = threading.Event()
        running.set()

        def _emit_status(message: str) -> None:
            print(f"[incremental/runtime] {message}", flush=True)

        def _emit_debug_event(row: dict[str, object]) -> None:
            nonlocal debug_event_count, last_progress_report
            debug_event_count += 1
            now = time.monotonic()
            if now - last_progress_report >= 5.0:
                last_progress_report = now
                current_elapsed = 0.0
                meta = row.get("meta")
                if isinstance(meta, dict):
                    try:
                        current_elapsed = float(meta.get("elapsed_seconds") or 0.0)
                    except Exception:
                        current_elapsed = 0.0
                if current_elapsed <= 0.0:
                    current_elapsed = float(debug_event_count) * float(getattr(cfg, "hop_seconds", 0.0) or 0.0)
                wall_elapsed = max(0.001, now - started)
                pct = (current_elapsed / replay_duration_seconds) if replay_duration_seconds > 0.0 else 0.0
                eta = (wall_elapsed * (1.0 - pct) / pct) if pct > 0.0 else 0.0
                speed = current_elapsed / wall_elapsed
                timing = meta.get("runtime_timing") if isinstance(meta, dict) else None
                timing_summary = ""
                if isinstance(timing, dict):
                    timing_summary = f"; timing {_format_runtime_timing(timing)}"
                print(
                    "[incremental] progress "
                    f"{_format_progress_bar(pct)} "
                    f"{pct * 100.0:5.1f}% window={debug_event_count}; "
                    f"audio={current_elapsed:.1f}/{replay_duration_seconds:.1f}s; "
                    f"speed={speed:.2f} audio-s/s; eta={eta:.1f}s"
                    f"{timing_summary}",
                    flush=True,
                )
            if trace_handle is None:
                return
            payload = dict(row)
            payload["window_index"] = int(debug_event_count)
            trace_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def _record_transcript_event(event: dict[str, object]) -> None:
            nonlocal final_text, accumulated_text, raw_hits, recorded_event_count, current_audio_elapsed
            raw = str(event.get("raw_text") or "")
            source = str(event.get("source_text") or "")
            translated = str(event.get("translated_text") or "")
            meta = event.get("meta")
            if isinstance(meta, dict):
                try:
                    current_audio_elapsed = float(meta.get("elapsed_seconds") or current_audio_elapsed)
                except Exception:
                    pass
            if raw.strip():
                raw_hits += 1
            if not source.strip():
                return
            final_text = _normalize_incremental_text(source)
            accumulated_text = _append_runtime_snapshot(accumulated_text, final_text)
            # Do not feed overlapping rolling-window token metadata into the
            # exporter. Realtime compare should export the final history
            # snapshot; per-window token rows remain available in
            # realtime_debug_trace.jsonl for diagnostics. Recording every
            # window here stacks the same audio region many times and produces
            # repeated CJK characters in realtime_project.*.
            del translated, meta
            recorded_event_count += 1

        def _emit_subtitle_ready(source_text: str, translated_text: str) -> None:
            del translated_text
            nonlocal last_overlay_snapshot_s
            # source_text here is the overlay frame (committed | separator | raw),
            # display-only. It must NOT feed final_text/accumulated_text (those are
            # owned by the clean _record_transcript_event source). Sample it on the
            # overlay-snapshot cadence instead.
            if not source_text.strip():
                return
            if float(overlay_snapshot_seconds) <= 0.0:
                return
            if (current_audio_elapsed - last_overlay_snapshot_s) >= float(overlay_snapshot_seconds):
                last_overlay_snapshot_s = current_audio_elapsed
                overlay_timeline.append((float(current_audio_elapsed), str(source_text)))

        deps = TranscriptionLoopDeps(
            config=cfg,
            subtitle_assembler=assembler,
            text_delta_logger=TextDeltaLogger(lambda _prefix, _text: None),
            segment_artifacts=SegmentArtifacts(log_dir=str(cfg.log_dir)),
            gpu_telemetry=GpuTelemetryReporter(interval_seconds=5.0),
            get_capture=lambda: capture,
            get_transcriber=lambda: transcriber,
            get_preprocess_pipeline=lambda: preprocess_pipeline,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda _message: False,
            emit_status=_emit_status,
            emit_debug_event=_emit_debug_event,
            emit_subtitle_ready=_emit_subtitle_ready,
            record_transcript_event=_record_transcript_event,
        )
        engine = TranscriptionLoopEngine(deps)
        engine.run(running)
        elapsed = time.monotonic() - started
        timing_summary = engine.get_timing_summary()
        print(
            "[incremental] completed "
            f"debug_events={debug_event_count}; recorded_events={recorded_event_count}; "
            f"raw_non_empty={raw_hits}; elapsed={elapsed:.1f}s; "
            f"realtime_factor={float(timing_summary.get('realtime_factor', 0.0) or 0.0):.3f}x; "
            f"dominant={timing_summary.get('dominant_stage', '')}",
            flush=True,
        )

        if overlay_timeline_path is not None and float(overlay_snapshot_seconds) > 0.0 and overlay_timeline:
            try:
                overlay_timeline_path.parent.mkdir(parents=True, exist_ok=True)
                blocks = [
                    f"=== audio t≈{audio_s:.1f}s ===\n{frame.rstrip()}"
                    for (audio_s, frame) in overlay_timeline
                ]
                header = (
                    f"# Live overlay frame timeline (every {float(overlay_snapshot_seconds):g}s of audio)\n"
                    "# Display-only: committed history | separator | live raw window (immediate speaker marker).\n"
                    f"# snapshots={len(overlay_timeline)}\n\n"
                )
                overlay_timeline_path.write_text(header + "\n\n".join(blocks) + "\n", encoding="utf-8")
                print(f"[incremental] overlay timeline: {overlay_timeline_path} ({len(overlay_timeline)} snapshots)", flush=True)
            except Exception as exc:
                print(f"[incremental] overlay timeline write failed: {exc}", flush=True)

        output_text = final_text or accumulated_text
        speaker_reconciliation = _reconcile_incremental_speaker_profiles(
            transcriber,
            threshold=float(max(0.0, speaker_profile_reconcile_threshold)),
        )
        del exporter
        return {
            "text": output_text,
            "window_count": int(debug_event_count),
            "raw_non_empty_windows": int(raw_hits),
            "speaker_profile_reconciliation": speaker_reconciliation,
            "timing_summary": timing_summary,
        }
    finally:
        if trace_handle is not None:
            trace_handle.close()
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        _dispose_transcriber(transcriber)
        _release_runtime_memory("incremental phase cleanup")

def _run_direct(
    cfg: RuntimeConfig,
    full_audio: AudioChunk,
    *,
    exporter: TranscriptExporterSession,
    require_gpu: bool,
    chunk_seconds: float,
    language_subchunk_seconds: float = 30.0,
    speaker_profile_reconcile_threshold: float = 0.0,
    whole_file_diarization: bool = True,
) -> dict[str, object]:
    transcriber = None
    beat_stop = threading.Event()
    beat_thread = None
    try:
        transcriber = create_stt_transcriber(cfg, progress_callback=_make_status_callback("direct/status"))
        print(f"[direct] transcriber: {_describe_transcriber(transcriber)}", flush=True)
        if require_gpu and (not str(getattr(transcriber, "_device", "")).lower().startswith("cuda")):
            raise RuntimeError("direct path is not on CUDA device; abort by --allow-cpu-fallback to continue.")

        def _heartbeat() -> None:
            started = time.monotonic()
            while not beat_stop.wait(8.0):
                elapsed = time.monotonic() - started
                print(f"[direct] running... elapsed={elapsed:.1f}s", flush=True)

        beat_thread = threading.Thread(target=_heartbeat, daemon=True)
        beat_thread.start()
        started = time.monotonic()
        duration = _audio_duration_seconds(full_audio)

        def _print_direct_progress(completed_audio_seconds: float, total_audio_seconds: float) -> None:
            if total_audio_seconds <= 0.0:
                return
            elapsed = max(0.001, time.monotonic() - started)
            completed = min(float(total_audio_seconds), max(0.0, float(completed_audio_seconds)))
            pct = completed / float(total_audio_seconds)
            eta = (elapsed * (1.0 - pct) / pct) if pct > 0.0 else 0.0
            speed = completed / elapsed
            print(
                "[direct] progress "
                f"{_format_progress_bar(pct)} "
                f"{pct * 100.0:5.1f}% audio={completed:.1f}/{total_audio_seconds:.1f}s; "
                f"speed={speed:.2f} audio-s/s; eta={eta:.1f}s",
                flush=True,
            )

        result = run_direct_transcription(
            cfg,
            full_audio,
            transcriber=transcriber,
            chunk_seconds=chunk_seconds,
            language_subchunk_seconds=language_subchunk_seconds,
            speaker_profile_reconcile_threshold=speaker_profile_reconcile_threshold,
            whole_file_diarization=whole_file_diarization,
            on_progress=_print_direct_progress,
            on_status=lambda message: print(f"[direct] {message}", flush=True),
        )
        text = str(result.get("text") or "")
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        if text or meta.get("token_timestamps"):
            exporter.record(raw_text=text, source_text=text, translated_text="", meta=meta)
        print(f"[direct] done elapsed={time.monotonic() - started:.1f}s chunks={meta.get('direct_chunk_count', 0)}", flush=True)
        return {"text": text, "meta": meta}
    finally:
        beat_stop.set()
        if beat_thread is not None:
            beat_thread.join(timeout=0.5)
        _dispose_transcriber(transcriber)
        _release_runtime_memory("direct phase cleanup")


_VOICE_BASENAME = "voice"
_TEXT_BASENAME = "text"
_AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm")
_TEXT_EXTS = (".txt", ".srt", ".vtt")


class CaseInput:
    """One compare case: an audio file plus an optional ground-truth subtitle.

    New input layout (one case per folder under input/):
        input/<case>/voice.<ext>   required audio (fixed basename "voice")
        input/<case>/text.<ext>    optional ground-truth subtitle (fixed basename "text")
    """

    def __init__(self, name: str, audio_path: Path, reference_path: Path | None = None) -> None:
        self.name = name
        self.audio_path = audio_path
        self.reference_path = reference_path


def _find_named_file(folder: Path, basename: str, exts: tuple[str, ...]) -> Path | None:
    for ext in exts:
        candidate = folder / f"{basename}{ext}"
        if candidate.is_file():
            return candidate
    # Fallback: any file whose stem matches the fixed basename (case-insensitive).
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.stem.lower() == basename.lower():
            return child
    return None


_SUBTITLE_EXT_PRIORITY = (".srt", ".vtt", ".txt")
_LANG_CODE_RE = re.compile(
    r"^(zh|cmn|yue|wuu|en|ja|ko|fr|de|es|ru|it|pt|vi|th|id|ms)(?:[-_][a-z]{2,4})?$",
    flags=re.IGNORECASE,
)


def _language_from_stem(stem: str) -> str | None:
    """If a subtitle filename looks like a language code (zh / zh-hant / en ...), return it."""
    s = str(stem or "").strip()
    return s.lower() if _LANG_CODE_RE.match(s) else None


def _find_reference_subtitle(folder: Path) -> Path | None:
    """Find a ground-truth subtitle in a case folder.

    The filename is not fixed: it is often the subtitle language (e.g. zh.srt, en.srt),
    but that is not guaranteed, and the file may be absent. Prefer .srt, then .vtt, then
    .txt; among equal extensions prefer a language-code filename. Falls back to the legacy
    fixed 'text' basename (with or without extension).
    """
    files = [p for p in sorted(folder.iterdir()) if p.is_file()]
    for ext in _SUBTITLE_EXT_PRIORITY:
        matches = [p for p in files if p.suffix.lower() == ext]
        if matches:
            lang_named = [p for p in matches if _language_from_stem(p.stem)]
            return (lang_named or matches)[0]
    for p in files:  # legacy: fixed 'text' basename with any/no extension
        if p.stem.lower() == _TEXT_BASENAME:
            return p
    return None


def _case_from_folder(folder: Path) -> CaseInput | None:
    voice = _find_named_file(folder, _VOICE_BASENAME, _AUDIO_EXTS)
    if voice is None:
        return None
    reference = _find_reference_subtitle(folder)
    return CaseInput(name=folder.name, audio_path=voice, reference_path=reference)


def _collect_cases(input_arg: str, *, input_dir: Path, all_files: bool) -> list[CaseInput]:
    # Explicit --input: a case folder (containing voice.*) or a single audio file (legacy).
    if input_arg.strip():
        path = Path(input_arg).resolve()
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        if path.is_dir():
            case = _case_from_folder(path)
            if case is None:
                raise FileNotFoundError(f"no '{_VOICE_BASENAME}.*' audio under case folder: {path}")
            return [case]
        return [CaseInput(name=path.stem, audio_path=path, reference_path=None)]

    if not input_dir.exists():
        raise FileNotFoundError(f"compare input directory not found: {input_dir}")

    # New layout: each subfolder under input/ is a case containing voice.* (+ optional text.*).
    case_folders = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    cases = [c for c in (_case_from_folder(folder) for folder in case_folders) if c is not None]
    if cases:
        return cases if all_files else [cases[0]]

    # Legacy fallback: loose audio files directly under input/.
    loose = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTS])
    if not loose:
        raise FileNotFoundError(
            f"no case folder with '{_VOICE_BASENAME}.*' and no loose audio file found under: {input_dir}"
        )
    selected = loose if all_files else loose[:1]
    return [CaseInput(name=p.stem, audio_path=p, reference_path=None) for p in selected]


def _read_reference_subtitle(path: Path) -> str:
    raw = _try_read_text(str(path))
    if not raw.strip():
        return ""
    looks_like_srt = path.suffix.lower() in {".srt", ".vtt"} or "-->" in raw
    return _strip_subtitle_timing(raw) if looks_like_srt else raw


def _strip_subtitle_timing(text: str) -> str:
    """Reduce an SRT/VTT subtitle to plain text lines.

    Drops cue indices and `-->` timestamp lines, WEBVTT headers, and NOTE/STYLE blocks,
    then strips inline `<...>` tags and `{...}` ASS-style overrides from cue text.
    """
    lines: list[str] = []
    for line in str(text or "").splitlines():
        s = line.strip()
        if not s or s.isdigit() or "-->" in s:
            continue
        up = s.upper()
        if up == "WEBVTT" or up.startswith("NOTE") or up.startswith("STYLE"):
            continue
        s = re.sub(r"<[^>]+>", "", s)        # <i>, <b>, <font ...>
        s = re.sub(r"\{[^}]*\}", "", s)      # {\an8} ASS-style overrides
        s = s.strip()
        if s:
            lines.append(s)
    return "\n".join(lines).strip()


def _strip_speaker_markers_text(text: str) -> str:
    """Remove leading [spk_xxx]/S0: style speaker markers, keeping text and line structure."""
    out: list[str] = []
    for raw in str(text or "").splitlines():
        _, body = _extract_project_txt_speaker_and_body(raw)
        if body:
            out.append(body)
    return "\n".join(out).strip()


def _write_reference_comparison(
    case_dir: Path,
    *,
    out_prefix: str,
    candidate_label: str,
    candidate_norm: str,
    reference_norm: str,
    language_hint: str | None,
    reference_path: Path,
) -> dict[str, object]:
    """Compare one candidate transcript (already normalized) against the ground-truth subtitle.

    Marker semantics (reference vs candidate): `.` same / `-` candidate-extra (candidate has,
    reference does not) / `+` candidate-missing (reference has, candidate dropped).
    """
    diff = _build_reference_diff(reference_norm, candidate_norm, language_hint=language_hint)
    distance = _levenshtein(candidate_norm, reference_norm)
    cer = float(distance) / float(max(1, len(reference_norm)))
    result = {
        "normalized_cer": float(cer),
        "normalized_char_distance": int(distance),
        "reference_chars": int(len(reference_norm)),
        "candidate_chars": int(len(candidate_norm)),
        "compare_unit": str(diff.get("compare_unit", "char")),
        "extra_units": int(diff.get("candidate_extra_count", 0)),
        "missing_units": int(diff.get("candidate_missing_count", 0)),
        "marker_line": str(diff.get("marker_line", "")),
    }
    (case_dir / f"{out_prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (case_dir / f"{out_prefix}.txt").write_text(
        "\n".join(
            [
                f"reference_file={reference_path}",
                f"candidate={candidate_label}",
                f"normalized_cer={cer:.6f}",
                f"compare_unit={result['compare_unit']}",
                f"extra_units={result['extra_units']}  ({candidate_label} has, reference does not)",
                f"missing_units={result['missing_units']}  (reference has, {candidate_label} dropped)",
                "",
                f"[diff_marker_line]  . same  - {candidate_label}-extra  + {candidate_label}-missing",
                str(diff.get("marker_line", "")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[ref] {candidate_label} vs reference: cer={cer:.4f} "
        f"extra={result['extra_units']} missing={result['missing_units']}",
        flush=True,
    )
    return result


def _safe_name(path: Path) -> str:
    return _safe_name_str(path.stem)


def _safe_name_str(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_") or "audio"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare direct WhisperX subtitles vs project incremental realtime subtitles using compare_whisperx_test audio."
    )
    parser.add_argument(
        "--input",
        default="",
        help="A case folder containing voice.* (+ optional text.*), or a single audio file (legacy). "
        "If empty, use --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Root of compare cases. Each subfolder is a case with voice.* and optional text.* ground-truth subtitle.",
    )
    parser.add_argument("--all-files", action="store_true", help="Run all files under --input-dir.")
    parser.add_argument("--model", default="", help="WhisperX model ref/path. Empty = persisted setting, then local stt cache.")
    parser.add_argument(
        "--model-root",
        default=str(SRC_ROOT / "models" / "whisperx" / "stt"),
        help="Local WhisperX STT model root; model aliases resolve under this folder first.",
    )
    parser.add_argument("--language", default="auto", help="Source language hint, e.g. zh/en/ja. auto = WhisperX auto-detect.")
    parser.add_argument(
        "--lock-realtime-language-from-direct",
        action="store_true",
        help=(
            "When --language auto, force realtime/file-replay to use the direct pass detected language. "
            "Default off keeps realtime closer to main runtime auto-language behavior."
        ),
    )
    parser.add_argument("--align-model", default="", help="WhisperX alignment model id/path (empty = keep runtime setting).")
    parser.add_argument(
        "--align-language",
        default="follow-source",
        help="WhisperX alignment language strategy/value (e.g. follow-source, en, zh).",
    )
    parser.add_argument("--align-device", choices=["auto", "cuda", "cpu"], default="auto", help="WhisperX alignment device override.")
    parser.add_argument(
        "--force-alignment",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override forced alignment switch. auto = keep profile/runtime setting.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda", help="STT device preference.")
    parser.add_argument("--compute-type", choices=["float16", "int8_float16", "int8"], default="float16", help="WhisperX ASR compute type.")
    parser.add_argument("--beam-size", type=int, default=5, help="WhisperX ASR beam size. 5 preserves WhisperX default; 1 is faster.")
    parser.add_argument("--batch-size", type=int, default=4, help="WhisperX ASR batch size.")
    parser.add_argument(
        "--realtime-stt-provider",
        choices=["whisperx", "whispercpp"],
        default="whisperx",
        help=(
            "STT provider for the realtime/incremental pass only. The direct pass always stays on "
            "WhisperX/CUDA (it is the accuracy yardstick both passes are scored against), regardless "
            "of this flag."
        ),
    )
    parser.add_argument(
        "--whispercpp-model-size",
        default="medium",
        help=(
            "whisper.cpp model size, used only when --realtime-stt-provider whispercpp. "
            "Round 0033 verified 'medium' as the only clean/non-duplicating realtime model; "
            "large-v2/large-v3 have known live-quality problems -- do not default to them "
            "without reading docs/history round 0033."
        ),
    )
    parser.add_argument(
        "--whispercpp-mode",
        choices=["server", "subprocess"],
        default="server",
        help=(
            "whisper.cpp execution mode, used only when --realtime-stt-provider whispercpp. "
            "Round 0033 shipped resident 'server' (whisper-server) as the only realtime-viable mode; "
            "'subprocess' reloads the model every window and is not realtime."
        ),
    )
    parser.add_argument("--preset", choices=["", "balanced", "high-accuracy"], default="", help="Apply a runtime preset bundle (round 0015) onto the realtime cfg, overriding model/compute/beam/seg-hop/alignment/diarization/speaker-profile.")
    parser.add_argument("--rolling-prompt-chars", type=int, default=0, help="Per-window rolling initial_prompt size in chars (0 disables). 0010/0012 diagnostic lever.")
    parser.add_argument("--display-script", choices=["off", "hant", "hans"], default="hant", help="Display-script fold for visible/exported subtitle (char-level, CER-neutral).")
    parser.add_argument(
        "--profile",
        choices=["fast", "accurate"],
        default="fast",
        help="Benchmark profile: fast=alignment on, diarization off; accurate=use persisted WhisperX switches.",
    )
    parser.add_argument(
        "--unsafe-cuda-align",
        action="store_true",
        help="Set VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN=1 for this run.",
    )
    parser.add_argument(
        "--align-guard",
        choices=["", "safe", "unsafe-cuda", "probe"],
        default="",
        help="Override whisperx_align_guard for this run. Empty keeps persisted setting. "
        "'probe' = per-language CUDA capability probe (GPU align where safe, CPU fallback otherwise).",
    )
    parser.add_argument(
        "--diarization-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Override whisperx_diarization_device for this run. auto = keep persisted setting.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow run to continue when transcriber falls back to CPU.",
    )
    parser.add_argument("--segment-seconds", type=float, default=9.6, help="Incremental segment seconds.")
    parser.add_argument("--hop-seconds", type=float, default=1.2, help="Incremental hop seconds.")
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.0,
        help="File-replay feed speed multiple: 0=unlimited (max, pegs GPU/RAM at 100%%), 1.0=realtime, "
        "<1 slower. Pace tests so each window gets ~1.5x its processing time (replay_speed ~= 1/(1.5*rtf): "
        "GPU-align zh ~1.5, CPU-align zh ~0.6) to keep the machine usable during long runs.",
    )
    parser.add_argument(
        "--speaker-profile-match-threshold",
        type=float,
        default=0.65,
        help="Accurate-profile speaker embedding match threshold for compare runs. <=0 keeps persisted runtime value.",
    )
    parser.add_argument(
        "--speaker-profile-min-seconds",
        type=float,
        default=2.0,
        help="Minimum speech seconds before creating/updating a speaker profile in compare runs. <=0 keeps persisted runtime value.",
    )
    parser.add_argument(
        "--speaker-profile-backend",
        default="",
        help=(
            "Round 0045 Fix 2: cross-window profile embedding backend "
            "(pyannote|wespeaker|speechbrain-ecapa|nemo-titanet). Empty keeps persisted value. "
            "'wespeaker' uses diar-3.1's own embedding (separates zh where pyannote/embedding collapses)."
        ),
    )
    parser.add_argument("--speaker-realtime-candidate-seconds", type=float, default=None,
                        help="Realtime rolling-window candidate maturity floor (seconds). Default keeps persisted/shipped 6.0. Direct chunks unaffected.")
    parser.add_argument("--speaker-realtime-candidate-samples", type=int, default=None,
                        help="Realtime candidate maturity floor (window count). Default keeps persisted/shipped 8.")
    parser.add_argument("--speaker-realtime-candidate-match-threshold", type=float, default=None,
                        help="Realtime candidate-match similarity gate (decoupled from match_threshold). "
                             "0.0/unset keeps legacy match_threshold-0.05; lower (e.g. 0.55) reduces candidate "
                             "fragmentation so minority speakers reach the promotion floor. Direct chunks unaffected.")
    parser.add_argument("--speaker-realtime-update-match-threshold", type=float, default=None,
                        help="Realtime centroid-UPDATE gate, decoupled from the assign gate. 0.0/unset = update at "
                             "assign (legacy); higher (e.g. 0.85) only blends a clip into a profile centroid on a "
                             "strong match, keeping centroids pure so the dominant profile cannot drift/absorb. "
                             "Direct chunks unaffected.")
    parser.add_argument("--speaker-realtime-visible-seconds", type=float, default=None,
                        help="Realtime visible-identity maturity floor (seconds). Default keeps persisted/shipped 24.0.")
    parser.add_argument("--speaker-realtime-visible-samples", type=int, default=None,
                        help="Realtime visible-identity maturity floor (window count). Default keeps persisted/shipped 16.")
    parser.add_argument("--speaker-realtime-refresh-seconds", type=float, default=None,
                        help="Forward-only speaker inventory refresh cadence/window in audio seconds. None keeps persisted; 0 disables.")
    parser.add_argument("--speaker-realtime-refresh-alpha", type=float, default=None,
                        help="Forward-only refresh EMA trust toward offline profile-space centroid. None keeps persisted.")
    parser.add_argument("--speaker-realtime-refresh-assign-threshold", type=float, default=None,
                        help="Forward-only refresh profile-to-cluster cosine floor. None keeps persisted.")
    parser.add_argument("--speaker-realtime-refresh-min-cluster-seconds", type=float, default=None,
                        help="Forward-only refresh sliver filter in seconds. None keeps persisted.")
    parser.add_argument("--speaker-realtime-refresh-merge", dest="speaker_realtime_refresh_merge", action="store_true", default=None,
                        help="Enable offline-arbitrated profile merge during refresh.")
    parser.add_argument("--no-speaker-realtime-refresh-merge", dest="speaker_realtime_refresh_merge", action="store_false",
                        help="Disable offline-arbitrated profile merge during refresh.")
    parser.add_argument("--speaker-count-hint", dest="speaker_count_hint_enabled", action="store_true", default=None,
                        help="Round 0055: enable automatic scalar speaker-count cap feedback. None keeps persisted/default off.")
    parser.add_argument("--no-speaker-count-hint", dest="speaker_count_hint_enabled", action="store_false",
                        help="Round 0055: disable automatic scalar speaker-count cap feedback.")
    parser.add_argument("--speaker-count-hint-seconds", type=float, default=None,
                        help="Round 0055: cadence between speaker-count estimation passes. None keeps persisted/default.")
    parser.add_argument("--speaker-count-hint-window-seconds", type=float, default=None,
                        help="Round 0055: bounded rolling audio window analyzed by count estimation. None keeps persisted/default.")
    parser.add_argument("--speaker-count-hint-sliver-floor-seconds", type=float, default=None,
                        help="Round 0055: drop clusters shorter than this before counting speakers. None keeps persisted/default.")
    parser.add_argument("--speaker-merge-grace-windows", type=int, default=None,
                        help="Round 0056: post-merge online-match grace window count. None keeps persisted/default off; 0 disables.")
    parser.add_argument("--speaker-merge-grace-relief", type=float, default=None,
                        help="Round 0056: temporary cosine-threshold relief for a graced merged profile. None keeps persisted/default.")
    parser.add_argument("--speaker-merge-preserve-centroid", dest="speaker_merge_preserve_centroid", action="store_true", default=None,
                        help="Round 0057: preserve the survivor centroid during profile merge. None keeps persisted/default off.")
    parser.add_argument("--no-speaker-merge-preserve-centroid", dest="speaker_merge_preserve_centroid", action="store_false",
                        help="Round 0057: use weighted-average centroid blending during profile merge.")
    parser.add_argument("--speaker-profile-max-exemplars", type=int, default=None,
                        help="Round 0061: bounded multi-exemplar profile representation. None/1 keeps single-centroid (byte-identical).")
    parser.add_argument("--speaker-profile-exemplar-diversity-threshold", type=float, default=None,
                        help="Round 0061: similarity threshold deciding blend-into-nearest-exemplar vs. add-a-new-exemplar. Only consulted when max-exemplars > 1.")
    parser.add_argument("--subtitle-relabel", dest="subtitle_relabel_enabled", action="store_true", default=None,
                        help="Round 0048: enable pre-commit local-diarization relabel for the live overlay. None keeps persisted; unset = disabled.")
    parser.add_argument("--no-subtitle-relabel", dest="subtitle_relabel_enabled", action="store_false",
                        help="Round 0048: disable pre-commit local-diarization relabel.")
    parser.add_argument("--subtitle-relabel-window-seconds", type=float, default=None,
                        help="Round 0048: local-diarization window / effective commit-hold for the pre-commit relabel. None keeps persisted.")
    parser.add_argument("--subtitle-relabel-sliver-floor-seconds", type=float, default=None,
                        help="Round 0048: sliver filter for the pre-commit relabel's local clusters. None keeps persisted.")
    parser.add_argument("--subtitle-relabel-assign-threshold", type=float, default=None,
                        help="Round 0048: profile-match cosine floor for the pre-commit relabel. None keeps persisted (this is the knob to sweep in the A/B).")
    parser.add_argument("--subtitle-relabel-margin", type=float, default=None,
                        help="Round 0052: turn-aware relabel overwrite margin. None keeps persisted/default 0.05.")
    parser.add_argument("--subtitle-relabel-async", dest="subtitle_relabel_async", action="store_true", default=None,
                        help="Round 0052 Phase B: resolve relabel spans on a background worker instead of the loop thread. None keeps persisted/default off.")
    parser.add_argument("--no-subtitle-relabel-async", dest="subtitle_relabel_async", action="store_false",
                        help="Round 0052 Phase B: force the synchronous relabel path.")
    parser.add_argument("--asr-temperatures", type=str, default=None,
                        help="Round 0049: comma-separated temperature-fallback schedule override (e.g. '0.0,0.2,0.4'). None keeps persisted; empty string also keeps library default.")
    parser.add_argument("--asr-log-prob-threshold", type=float, default=None,
                        help="Round 0049: fallback-trigger override (library default -1.0). None keeps persisted/default.")
    parser.add_argument("--asr-compression-ratio-threshold", type=float, default=None,
                        help="Round 0049: fallback-trigger override (library default 2.4). None keeps persisted/default.")
    parser.add_argument("--asr-no-speech-threshold", type=float, default=None,
                        help="Round 0049: fallback-trigger override (library default 0.6). None keeps persisted/default.")
    parser.add_argument("--diarization-min-speakers", type=int, default=None,
                        help="Round 0054: optional pyannote min_speakers hint. None keeps persisted/default; 0=auto.")
    parser.add_argument("--diarization-max-speakers", type=int, default=None,
                        help="Round 0054: optional pyannote max_speakers hint and online profile cap. None keeps persisted/default; 0=auto.")
    parser.add_argument(
        "--speaker-profile-reconcile-threshold",
        type=float,
        default=0.52,
        help=(
            "Final session-level speaker profile merge threshold after direct/realtime pass. "
            "0 uses provider default. Lower values merge more aggressively."
        ),
    )
    parser.add_argument(
        "--speaker-profile-quality-gate",
        dest="speaker_profile_quality_gate",
        action="store_true",
        help=(
            "Round 0023: gate the speaker-profile learn path. Low-quality clips (gibberish/music/low-confidence) "
            "can still match an existing profile for display but never update/create a centroid. "
            "A/B this against the default (gate off) — CER must stay byte-identical; the win is fewer spurious speakers."
        ),
    )
    parser.add_argument(
        "--speaker-label-max-speakers",
        type=int,
        default=0,
        help="Optional compare/export display cap for speaker labels. 0 keeps all detected profile speakers.",
    )
    parser.add_argument("--overlay-snapshot-seconds", type=float, default=30.0, help="Round 0017: snapshot the live overlay frame (committed|raw boundary) every N seconds of audio to realtime_overlay_timeline.txt. 0 disables.")
    parser.add_argument("--subtitle-commit-hold-seconds", type=float, default=0.0, help="Round 0017 delayed-freeze: hold committed batches N seconds so speaker markers can be re-anchored/back-dated (realtime path only). 0=off.")
    parser.add_argument("--export-formats", default="txt,srt,json", help="Export formats: txt,srt,json")
    parser.add_argument("--no-export-timestamps", action="store_true", help="Exclude timestamps from txt/json export.")
    parser.add_argument("--no-export-speaker", action="store_true", help="Exclude speaker labels from export.")
    parser.add_argument(
        "--direct-group-seconds",
        type=float,
        default=0.0,
        help="For compare text only: regroup direct transcript by fixed seconds (e.g. 30). 0=off.",
    )
    parser.add_argument(
        "--direct-chunk-seconds",
        type=float,
        default=30.0,
        help=(
            "Transcribe direct WhisperX reference in project-side chunks. "
            "0=single full-file pass; positive values select project-side chunk seconds."
        ),
    )
    parser.add_argument(
        "--direct-language-subchunk-seconds",
        type=float,
        default=30.0,
        help=(
            "When --language auto and --direct-chunk-seconds is larger than this value, "
            "split each direct chunk into language-routing subchunks. 0=disable."
        ),
    )
    parser.add_argument(
        "--direct-whole-file-diarization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Round 0045: run direct diarization ONCE on the whole file (globally consistent "
            "labels, no profile re-cluster) instead of per-chunk. --no-direct-whole-file-diarization "
            "restores the legacy per-chunk diarization + profile reconcile."
        ),
    )
    parser.add_argument(
        "--realtime-compare-one-line",
        action="store_true",
        help="For compare text only: flatten realtime transcript into one line before normalization.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output root directory. Empty = app/src/tests/compare_whisperx_test/output/<timestamp>",
    )
    parser.add_argument(
        "--no-realtime-debug-trace",
        action="store_true",
        help="Disable per-window realtime trace export (raw/merged/history/stable/partial).",
    )
    args = parser.parse_args()

    input_dir = Path(str(args.input_dir or "")).resolve()
    try:
        cases = _collect_cases(args.input, input_dir=input_dir, all_files=bool(args.all_files))
    except Exception as exc:
        print(f"[error] {exc}", flush=True)
        return 2

    base_cfg = _load_base_cfg()
    language_arg = str(args.language or "").strip().lower()
    language = None if language_arg in {"", "auto"} else language_arg

    model_ref = _resolve_model_ref(
        model_arg=str(args.model or ""),
        model_root=Path(args.model_root).resolve(),
        persisted_model=str(getattr(base_cfg, "stt_model_path", "") or ""),
    )
    if bool(args.unsafe_cuda_align):
        os.environ["VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN"] = "1"
    cfg = _build_runtime_cfg(
        base_cfg,
        model_ref=model_ref,
        language=language,
        segment_seconds=float(max(1.0, args.segment_seconds)),
        hop_seconds=float(max(0.1, args.hop_seconds)),
        device=str(args.device),
        profile=str(args.profile),
        align_model=str(args.align_model or ""),
        align_language=str(args.align_language or "follow-source"),
        align_device=str(args.align_device),
        force_alignment=str(args.force_alignment),
        compute_type=str(args.compute_type),
        beam_size=int(args.beam_size),
        batch_size=int(args.batch_size),
        speaker_profile_match_threshold=float(args.speaker_profile_match_threshold),
        speaker_profile_min_seconds=float(args.speaker_profile_min_seconds),
        speaker_profile_reconcile_threshold=float(args.speaker_profile_reconcile_threshold),
        speaker_realtime_refresh_seconds=getattr(args, "speaker_realtime_refresh_seconds", None),
        speaker_realtime_refresh_alpha=getattr(args, "speaker_realtime_refresh_alpha", None),
        speaker_realtime_refresh_assign_threshold=getattr(args, "speaker_realtime_refresh_assign_threshold", None),
        speaker_realtime_refresh_min_cluster_seconds=getattr(args, "speaker_realtime_refresh_min_cluster_seconds", None),
        speaker_realtime_refresh_merge=getattr(args, "speaker_realtime_refresh_merge", None),
        speaker_count_hint_enabled=getattr(args, "speaker_count_hint_enabled", None),
        speaker_count_hint_seconds=getattr(args, "speaker_count_hint_seconds", None),
        speaker_count_hint_window_seconds=getattr(args, "speaker_count_hint_window_seconds", None),
        speaker_count_hint_sliver_floor_seconds=getattr(args, "speaker_count_hint_sliver_floor_seconds", None),
        speaker_merge_grace_windows=getattr(args, "speaker_merge_grace_windows", None),
        speaker_merge_grace_relief=getattr(args, "speaker_merge_grace_relief", None),
        speaker_merge_preserve_centroid=getattr(args, "speaker_merge_preserve_centroid", None),
        speaker_profile_max_exemplars=getattr(args, "speaker_profile_max_exemplars", None),
        speaker_profile_exemplar_diversity_threshold=getattr(args, "speaker_profile_exemplar_diversity_threshold", None),
        subtitle_relabel_enabled=getattr(args, "subtitle_relabel_enabled", None),
        subtitle_relabel_window_seconds=getattr(args, "subtitle_relabel_window_seconds", None),
        subtitle_relabel_sliver_floor_seconds=getattr(args, "subtitle_relabel_sliver_floor_seconds", None),
        subtitle_relabel_assign_threshold=getattr(args, "subtitle_relabel_assign_threshold", None),
        subtitle_relabel_margin=getattr(args, "subtitle_relabel_margin", None),
        subtitle_relabel_async=getattr(args, "subtitle_relabel_async", None),
        asr_temperatures=getattr(args, "asr_temperatures", None),
        asr_log_prob_threshold=getattr(args, "asr_log_prob_threshold", None),
        asr_compression_ratio_threshold=getattr(args, "asr_compression_ratio_threshold", None),
        asr_no_speech_threshold=getattr(args, "asr_no_speech_threshold", None),
        diarization_min_speakers=getattr(args, "diarization_min_speakers", None),
        diarization_max_speakers=getattr(args, "diarization_max_speakers", None),
        speaker_profile_quality_gate=bool(getattr(args, "speaker_profile_quality_gate", False)),
    )
    if str(getattr(args, "align_guard", "") or "").strip():
        cfg.whisperx_align_guard = str(args.align_guard).strip()
    if str(getattr(args, "diarization_device", "auto") or "auto") != "auto":
        cfg.whisperx_diarization_device = str(args.diarization_device)
    if str(getattr(args, "speaker_profile_backend", "") or "").strip():
        cfg.whisperx_speaker_profile_backend = str(args.speaker_profile_backend).strip()
    cfg.whisperx_rolling_prompt_chars = max(0, int(getattr(args, "rolling_prompt_chars", 0) or 0))
    _hold = float(getattr(args, "subtitle_commit_hold_seconds", 0.0) or 0.0)
    if _hold > 0.0:
        cfg.subtitle_commit_hold_seconds = _hold
    for _arg, _key in (
        ("speaker_realtime_candidate_seconds", "whisperx_speaker_realtime_candidate_seconds"),
        ("speaker_realtime_candidate_samples", "whisperx_speaker_realtime_candidate_samples"),
        ("speaker_realtime_candidate_match_threshold", "whisperx_speaker_realtime_candidate_match_threshold"),
        ("speaker_realtime_update_match_threshold", "whisperx_speaker_realtime_update_match_threshold"),
        ("speaker_realtime_visible_seconds", "whisperx_speaker_realtime_visible_seconds"),
        ("speaker_realtime_visible_samples", "whisperx_speaker_realtime_visible_samples"),
        ("speaker_realtime_refresh_seconds", "whisperx_speaker_realtime_refresh_seconds"),
        ("speaker_realtime_refresh_alpha", "whisperx_speaker_realtime_refresh_alpha"),
        ("speaker_realtime_refresh_assign_threshold", "whisperx_speaker_realtime_refresh_assign_threshold"),
        ("speaker_realtime_refresh_min_cluster_seconds", "whisperx_speaker_realtime_refresh_min_cluster_seconds"),
    ):
        _val = getattr(args, _arg, None)
        if _val is not None:
            setattr(cfg, _key, _val)
    cfg.subtitle_display_script = "" if str(getattr(args, "display_script", "hant")) == "off" else str(getattr(args, "display_script", "hant"))
    if str(getattr(args, "preset", "") or ""):
        from voice2text.settings.presets import apply_preset
        apply_preset(cfg, args.preset)
        cfg.stt_model_path = _resolve_model_ref(
            model_arg=str(cfg.model_size or ""),
            model_root=Path(args.model_root).resolve(),
            persisted_model="",
        )
        print(f"[preset] applied {args.preset}: model={cfg.stt_model_path or cfg.model_size}; compute={cfg.compute_type}; beam={cfg.whisper_beam_size}; seg/hop={cfg.segment_seconds}/{cfg.hop_seconds}; align={cfg.whisperx_enable_forced_alignment}; diar={cfg.whisperx_enable_diarization}; spk_profile={cfg.whisperx_speaker_profile_enabled}", flush=True)
    # Realtime-only STT provider swap (diagnostic A/B: whisper.cpp Vulkan vs WhisperX on the
    # incremental/live path). The direct yardstick pass is forced back onto WhisperX/CUDA
    # separately below (direct_cfg), regardless of this flag. stt_variant/model_device/compute_type
    # are left untouched: stt_variant stays "gpu" so whisper.cpp's own device resolution
    # (resolve_whispercpp_device) auto-picks Vulkan, and model_device/compute_type are
    # WhisperX-specific fields whisper.cpp ignores.
    cfg.stt_provider = str(args.realtime_stt_provider)
    if str(args.realtime_stt_provider) == "whispercpp":
        cfg.stt_whispercpp_model_size = str(args.whispercpp_model_size)
        cfg.stt_whispercpp_mode = str(args.whispercpp_mode)
        # cfg.stt_model_path is a resolved WhisperX model path (models/whisperx/stt/<model>),
        # set unconditionally above via _build_runtime_cfg. voice2text.stt.whispercpp_runtime.
        # resolve_whispercpp_model() falls back to stt_model_path as an "explicit" ggml model
        # path/dir whenever stt_whispercpp_model_path is unset, so leaving it populated makes
        # whisper.cpp look for ggml-<size>.bin under the WRONG (WhisperX) model directory and
        # fail with "whisper.cpp ggml model not found". Clear it so resolve_whispercpp_model()
        # falls through to its own default models/whispercpp/ + stt_whispercpp_model_size lookup.
        cfg.stt_model_path = ""
    require_gpu = (str(args.device) == "cuda") and (not bool(args.allow_cpu_fallback))
    print(
        "[config] "
        + f"model={cfg.stt_model_path or cfg.model_size}; "
        + f"language={cfg.source_language or 'auto'}; "
        + f"align_model={str(getattr(cfg, 'whisperx_alignment_model', '') or 'auto')}; "
        + f"align_language={str(getattr(cfg, 'whisperx_alignment_language', '') or 'auto')}; "
        + f"align_device={str(getattr(cfg, 'whisperx_alignment_device', 'auto') or 'auto')}; "
        + f"device={cfg.model_device}; "
        + f"compute_type={cfg.compute_type}; "
        + f"beam_size={cfg.whisper_beam_size}; "
        + f"batch_size={cfg.whisper_batch_size}; "
        + f"require_gpu={require_gpu}; "
        + f"profile={args.profile}; "
        + f"diarization={bool(getattr(cfg, 'whisperx_enable_diarization', False))}; "
        + f"diarization_device={str(getattr(cfg, 'whisperx_diarization_device', 'auto') or 'auto')}; "
        + f"forced_alignment={bool(getattr(cfg, 'whisperx_enable_forced_alignment', False))}; "
        + f"vad={bool(getattr(cfg, 'whisperx_enable_vad', False))}/{str(getattr(cfg, 'whisperx_vad_method', ''))}",
        flush=True,
    )

    formats = _parse_formats(args.export_formats)
    include_timestamps = not bool(args.no_export_timestamps)
    include_speaker = not bool(args.no_export_speaker)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.output_dir or "").strip():
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = (DEFAULT_OUTPUT_ROOT / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path, restore_logger = _install_run_logger(out_root)
    print(f"[log] run log: {log_path}", flush=True)
    print(f"[run] started_at={datetime.now().isoformat(timespec='seconds')}", flush=True)

    batch_results: list[dict[str, object]] = []
    success_rows: list[dict[str, object]] = []
    try:
        for idx, case in enumerate(cases, start=1):
            if idx > 1:
                _release_runtime_memory("between-case preflight cleanup")
            input_path = case.audio_path
            ref_note = f"; ref={case.reference_path.name}" if case.reference_path is not None else "; ref=none"
            print(f"[run] ({idx}/{len(cases)}) case={case.name}; audio={input_path.name}{ref_note}", flush=True)
            case_dir = out_root / _safe_name_str(case.name)
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                decoded_wav = _decode_to_wav_16k_mono(input_path, ffmpeg_dir=str(getattr(cfg, "ffmpeg_dll_dir", "") or ""))
                full_audio = _read_wav(decoded_wav)
                print(f"[mem] case-start: {_format_memory_snapshot()}", flush=True)

                direct_cfg = RuntimeConfig(**cfg.__dict__)
                # Direct is the accuracy yardstick and must always run WhisperX/CUDA, even when
                # --realtime-stt-provider whispercpp swapped the realtime cfg above.
                direct_cfg.stt_provider = "whisperx"
                direct_cfg.stt_variant = "gpu"
                direct_cfg.model_device = "cuda"
                direct_cfg.compute_type = "float16"
                direct_cfg.whisperx_speaker_profile_store_path = _prepare_case_speaker_profile_path(case_dir, "direct")
                direct_session = _create_export_session(
                    out_dir=case_dir / "_tmp_direct",
                    formats=formats,
                    include_timestamps=include_timestamps,
                    include_speaker=include_speaker,
                )
                direct = _run_direct(
                    direct_cfg,
                    full_audio,
                    exporter=direct_session,
                    require_gpu=require_gpu,
                    chunk_seconds=float(args.direct_chunk_seconds),
                    language_subchunk_seconds=float(args.direct_language_subchunk_seconds),
                    speaker_profile_reconcile_threshold=float(args.speaker_profile_reconcile_threshold),
                    whole_file_diarization=bool(args.direct_whole_file_diarization),
                )
                direct_exports = _finalize_and_rename_exports(direct_session, target_dir=case_dir, prefix="direct_whisperx")
                direct_speaker_label_normalization = _normalize_exported_speaker_labels(
                    direct_exports,
                    profile_path=direct_cfg.whisperx_speaker_profile_store_path,
                    max_speakers=int(args.speaker_label_max_speakers),
                    profile_remap=_speaker_reconcile_remap(
                        (direct.get("meta") if isinstance(direct.get("meta"), dict) else {}).get(
                            "speaker_profile_reconciliation",
                            {},
                        )
                    ),
                )
                direct_speaker_profile_diagnostics = _speaker_profile_diagnostics(
                    direct_cfg.whisperx_speaker_profile_store_path
                )
                shutil.rmtree(case_dir / "_tmp_direct", ignore_errors=True)
                direct_txt = _try_read_text(str(direct_exports.get("txt", "")))
                if not direct_txt.strip():
                    direct_txt = str(direct.get("text") or "")
                direct_text_for_compare = _project_txt_to_single_line(direct_txt)
                direct_text_for_html = _project_txt_to_compare_lines(direct_txt, include_speaker=True)
                if not direct_text_for_compare:
                    direct_text_for_compare = re.sub(r"\s+", " ", str(direct.get("text") or "")).strip()
                if not direct_text_for_html:
                    direct_text_for_html = str(direct.get("text") or "").strip()
                direct_meta = direct.get("meta")
                if not isinstance(direct_meta, dict):
                    direct_meta = {}

                # WhisperX no-speaker output + optional ground-truth subtitle comparison.
                # The reference (when present) is compared against BOTH the direct WhisperX
                # transcript here and the realtime (main) subtitle later in this case.
                direct_nospk_text = _strip_speaker_markers_text(direct_txt)
                if not direct_nospk_text:
                    direct_nospk_text = re.sub(r"\s+", " ", str(direct.get("text") or "")).strip()
                (case_dir / "direct_whisperx_nospk.txt").write_text(direct_nospk_text + "\n", encoding="utf-8")
                vs_reference: dict[str, object] | None = None
                reference_norm: str | None = None
                reference_lang: str | None = None
                reference_section_html: str = ""
                reference_text: str = ""  # hoisted: also feeds the interactive compare.html srt source
                if case.reference_path is not None:
                    reference_text = _read_reference_subtitle(case.reference_path)
                    if not reference_text.strip():
                        print(f"[ref] reference subtitle is empty/unreadable: {case.reference_path}", flush=True)
                    else:
                        reference_norm = _normalize_for_compare(reference_text)
                        reference_lang = (
                            str(direct_meta.get("detected_language") or "").strip().lower()
                            or _language_from_stem(case.reference_path.stem)
                            or str(cfg.source_language or "").strip().lower()
                            or None
                        )
                        (case_dir / "reference.txt").write_text(reference_text.strip() + "\n", encoding="utf-8")
                        direct_ref = _write_reference_comparison(
                            case_dir,
                            out_prefix="compare_vs_reference_direct",
                            candidate_label="whisperx",
                            candidate_norm=_normalize_for_compare(direct_nospk_text),
                            reference_norm=reference_norm,
                            language_hint=reference_lang,
                            reference_path=case.reference_path,
                        )
                        vs_reference = {
                            "reference_file": str(case.reference_path),
                            "reference_language": reference_lang or "",
                            "compare_unit": str(direct_ref.get("compare_unit", "char")),
                            "direct": direct_ref,
                        }
                        # Build the "SRT vs WhisperX" section for compare.html (newline-preserved).
                        ref_html_norm = _normalize_for_html_compare(reference_text)
                        direct_html_norm = _normalize_for_html_compare(direct_nospk_text)
                        ref_html_diff = _build_reference_diff(
                            ref_html_norm,
                            direct_html_norm,
                            language_hint=reference_lang,
                            preserve_newlines=True,
                        )
                        reference_section_html = "\n".join(
                            _render_diff_compare_section(
                                title="SRT vs WhisperX",
                                meta_line=(
                                    f"reference={html_lib.escape(case.reference_path.name)} | "
                                    f"unit={html_lib.escape(str(direct_ref.get('compare_unit', 'char')))} | "
                                    f"normalized_distance={int(direct_ref.get('normalized_char_distance', 0))} | "
                                    f"normalized_ratio={float(direct_ref.get('normalized_cer', 0.0)):.6f}"
                                ),
                                reference_label="SRT Reference (ground truth)",
                                reference_text=str(ref_html_diff.get("reference_aligned", "")),
                                candidate_label="WhisperX",
                                candidate_annotated_html=str(ref_html_diff.get("realtime_annotated_html", "")),
                                marker_line=str(direct_ref.get("marker_line", "")),
                                marker_legend=". = same | - = whisperx extra (vs srt) | + = whisperx missing (srt dropped)",
                            )
                        )

                direct_summary = {
                    "window_count": int(direct_meta.get("direct_chunk_count") or 1),
                    "raw_non_empty_windows": 1 if str(direct.get("text") or "").strip() else 0,
                    "char_count": len(str(direct.get("text") or "")),
                    "audio_duration_seconds": float(direct_meta.get("audio_duration_seconds") or 0.0),
                    "direct_chunk_seconds": float(direct_meta.get("direct_chunk_seconds") or 0.0),
                    "direct_chunk_mode": str(direct_meta.get("direct_chunk_mode") or ""),
                    "direct_auto_chunked": bool(direct_meta.get("direct_auto_chunked", False)),
                    "direct_requested_chunk_seconds": float(direct_meta.get("direct_requested_chunk_seconds") or 0.0),
                    "direct_language_subchunk_seconds": float(
                        direct_meta.get("direct_language_subchunk_seconds") or 0.0
                    ),
                }
                direct_detected_lang = str(
                    direct_meta.get("detected_language")
                    or direct_meta.get("language")
                    or ""
                ).strip().lower()
                del direct
                del direct_session
                del full_audio
                _release_runtime_memory("after direct export")

                incremental_cfg = RuntimeConfig(**cfg.__dict__)
                incremental_cfg.source_file_path = str(input_path)
                incremental_cfg.whisperx_speaker_profile_store_path = _prepare_case_speaker_profile_path(case_dir, "incremental")
                if (
                    bool(args.lock_realtime_language_from_direct)
                    and not str(incremental_cfg.source_language or "").strip()
                    and direct_detected_lang
                ):
                    incremental_cfg.source_language = direct_detected_lang
                    print(
                        f"[compare] lock incremental language from direct pass: {direct_detected_lang}",
                        flush=True,
                    )
                realtime_session = _create_export_session(
                    out_dir=case_dir / "_tmp_realtime",
                    formats=formats,
                    include_timestamps=include_timestamps,
                    include_speaker=include_speaker,
                )
                trace_path = None if bool(args.no_realtime_debug_trace) else (case_dir / "realtime_debug_trace.jsonl")
                incremental = _run_incremental(
                    incremental_cfg,
                    AudioChunk(pcm16=b"", sample_rate=16000, channels=1),
                    exporter=realtime_session,
                    require_gpu=require_gpu,
                    debug_trace_path=trace_path,
                    speaker_profile_reconcile_threshold=float(args.speaker_profile_reconcile_threshold),
                    overlay_snapshot_seconds=float(args.overlay_snapshot_seconds),
                    overlay_timeline_path=(case_dir / "realtime_overlay_timeline.txt"),
                    replay_speed=float(args.replay_speed),
                )
                realtime_payload_text = str(incremental.get("text") or "").strip()
                realtime_exports = _write_realtime_main_payload_exports(
                    target_dir=case_dir,
                    prefix="realtime_project",
                    formats=formats,
                    text=realtime_payload_text,
                    duration_seconds=float(direct_summary.get("audio_duration_seconds") or 0.0),
                )
                effective_realtime_speaker_label_max = _resolve_realtime_speaker_label_cap(
                    requested_max_speakers=int(args.speaker_label_max_speakers),
                    direct_text_for_compare=direct_text_for_compare,
                    profile=str(args.profile),
                )
                realtime_speaker_label_normalization = {
                    "disabled": True,
                    "reason": "realtime artifacts mirror main overlay payload; no speaker remap/coalesce post-process applied",
                    "max_speakers": int(effective_realtime_speaker_label_max),
                    "profile_remap_count": 0,
                    "changed_files": [],
                }
                realtime_speaker_profile_diagnostics = _speaker_profile_diagnostics(
                    incremental_cfg.whisperx_speaker_profile_store_path
                )
                shutil.rmtree(case_dir / "_tmp_realtime", ignore_errors=True)
                realtime_txt = _try_read_text(str(realtime_exports.get("txt", "")))
                if not realtime_txt.strip():
                    realtime_txt = str(incremental.get("text") or "")
                realtime_text_for_compare = _project_txt_to_single_line(realtime_txt)
                realtime_text_for_html = _project_txt_to_compare_lines(realtime_txt, include_speaker=True)
                if not realtime_text_for_compare:
                    realtime_text_for_compare = re.sub(r"\s+", " ", str(incremental.get("text") or "")).strip()
                if not realtime_text_for_html:
                    realtime_text_for_html = str(incremental.get("text") or "").strip()
                # Always materialize the marker-stripped realtime text (the interactive
                # compare.html "沒 spk" source needs it even when there is no srt reference).
                realtime_nospk_text = _strip_speaker_markers_text(realtime_txt)
                if not realtime_nospk_text:
                    realtime_nospk_text = re.sub(r"\s+", " ", str(realtime_text_for_compare or "")).strip()
                (case_dir / "realtime_project_nospk.txt").write_text(realtime_nospk_text + "\n", encoding="utf-8")
                incremental_summary = {
                    "window_count": int(incremental.get("window_count") or 0),
                    "raw_non_empty_windows": int(incremental.get("raw_non_empty_windows") or 0),
                    "char_count": len(str(incremental.get("text") or "")),
                    "speaker_profile_reconciliation": incremental.get("speaker_profile_reconciliation", {}),
                    "timing_summary": incremental.get("timing_summary", {}),
                }
                del incremental
                del realtime_session
                _release_runtime_memory("after realtime export")

                # Keep prior normalization for distance metric continuity.
                inc_norm = _normalize_for_compare(realtime_text_for_compare)

                # Realtime (main) subtitle vs the same ground-truth subtitle as direct.
                if reference_norm is not None and isinstance(vs_reference, dict):
                    vs_reference["realtime"] = _write_reference_comparison(
                        case_dir,
                        out_prefix="compare_vs_reference_realtime",
                        candidate_label="realtime",
                        candidate_norm=inc_norm,
                        reference_norm=reference_norm,
                        language_hint=reference_lang,
                        reference_path=case.reference_path,
                    )

                dir_norm = _normalize_for_compare(direct_text_for_compare)
                distance = _levenshtein(inc_norm, dir_norm)
                base_len = max(1, len(dir_norm))
                cer = float(distance) / float(base_len)
                language_hint = direct_detected_lang or str(cfg.source_language or "").strip().lower() or None
                diff_view = _build_reference_diff(
                    dir_norm,
                    inc_norm,
                    language_hint=language_hint,
                )
                html_diff_view = _build_reference_diff(
                    _normalize_for_html_compare(direct_text_for_html),
                    _normalize_for_html_compare(realtime_text_for_html),
                    language_hint=language_hint,
                    preserve_newlines=True,
                )
                if _should_compare_speakers(str(args.profile)):
                    speaker_compare = _speaker_compare_summary(direct_text_for_compare, realtime_text_for_compare)
                else:
                    speaker_compare = _speaker_compare_disabled_summary(str(args.profile))
            except Exception as exc:  # noqa: BLE001
                fail_row = {
                    "input": str(input_path),
                    "case_dir": str(case_dir),
                    "status": "failed",
                    "error": str(exc),
                }
                batch_results.append(fail_row)
                (case_dir / "compare_failed.txt").write_text(str(exc) + "\n", encoding="utf-8")
                print(f"[fail] {input_path}: {exc}", flush=True)
                _release_runtime_memory("case-failed cleanup")
                continue

            spk_vs_truth = _speaker_accuracy_vs_truth(case_dir, str(input_path))
            report = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "input": str(input_path),
                "decoded_wav": str(decoded_wav),
                **({"spk_vs_truth": spk_vs_truth} if spk_vs_truth else {}),
                "config": {
                    "model": cfg.stt_model_path or cfg.model_size,
                    "language": cfg.source_language or "auto",
                    "incremental_language_lock": direct_detected_lang
                    if bool(args.lock_realtime_language_from_direct)
                    else "",
                    "lock_realtime_language_from_direct": bool(args.lock_realtime_language_from_direct),
                    "align_model": str(getattr(cfg, "whisperx_alignment_model", "") or "auto"),
                    "align_language": str(getattr(cfg, "whisperx_alignment_language", "") or "auto"),
                    "align_device": str(getattr(cfg, "whisperx_alignment_device", "auto") or "auto"),
                    "device": cfg.model_device,
                    "require_gpu": bool(require_gpu),
                    "profile": str(args.profile),
                    "diarization_device": str(getattr(cfg, "whisperx_diarization_device", "auto") or "auto"),
                    "segment_seconds": cfg.segment_seconds,
                    "hop_seconds": cfg.hop_seconds,
                    "speaker_profile_match_threshold": float(cfg.whisperx_speaker_profile_match_threshold),
                    "speaker_profile_min_seconds": float(cfg.whisperx_speaker_profile_min_seconds),
                    "speaker_profile_reconcile_threshold": float(args.speaker_profile_reconcile_threshold),
                    "speaker_label_max_speakers": int(args.speaker_label_max_speakers),
                    "effective_realtime_speaker_label_max_speakers": int(effective_realtime_speaker_label_max),
                    "direct_group_seconds": float(args.direct_group_seconds),
                    "direct_chunk_seconds": float(direct_summary.get("direct_chunk_seconds") or 0.0),
                    "direct_requested_chunk_seconds": float(args.direct_chunk_seconds),
                    "direct_language_subchunk_seconds": float(
                        direct_summary.get("direct_language_subchunk_seconds") or 0.0
                    ),
                    "direct_chunk_mode": str(direct_summary.get("direct_chunk_mode") or ""),
                    "direct_auto_chunked": bool(direct_summary.get("direct_auto_chunked", False)),
                    "realtime_compare_one_line": bool(args.realtime_compare_one_line),
                    "formats": formats,
                    "include_timestamps": include_timestamps,
                    "include_speaker": include_speaker,
                },
                "incremental": incremental_summary,
                "direct": direct_summary,
                "exports": {
                    "direct_whisperx": direct_exports,
                    "realtime_project": realtime_exports,
                    "compare_html": str(case_dir / "compare.html"),
                },
                "metrics": {
                    "normalized_char_distance": int(distance),
                    "normalized_cer": float(cer),
                    "incremental_char_count": int(len(inc_norm)),
                    "direct_char_count": int(len(dir_norm)),
                    "compare_unit": str(diff_view.get("compare_unit", "char")),
                    "same_char_count": int(diff_view.get("same_count", 0)),
                    "realtime_extra_char_count": int(diff_view.get("candidate_extra_count", 0)),
                    "realtime_missing_char_count": int(diff_view.get("candidate_missing_count", 0)),
                    "same_unit_count": int(diff_view.get("same_count", 0)),
                    "realtime_extra_unit_count": int(diff_view.get("candidate_extra_count", 0)),
                    "realtime_missing_unit_count": int(diff_view.get("candidate_missing_count", 0)),
                },
                "speaker_compare": speaker_compare,
                "speaker_label_normalization": {
                    "direct": direct_speaker_label_normalization,
                    "realtime": realtime_speaker_label_normalization,
                    "direct_reconciliation": direct_meta.get("speaker_profile_reconciliation", {}),
                    "realtime_reconciliation": incremental_summary.get("speaker_profile_reconciliation", {}),
                },
                "speaker_profile_diagnostics": {
                    "direct": direct_speaker_profile_diagnostics,
                    "realtime": realtime_speaker_profile_diagnostics,
                },
                "compare_inputs": {
                    "direct_text_for_compare": direct_text_for_compare,
                    "realtime_text_for_compare": realtime_text_for_compare,
                },
                "diff_view": {
                    "compare_unit": str(diff_view.get("compare_unit", "char")),
                    "reference_aligned": str(diff_view.get("reference_aligned", "")),
                    "realtime_aligned": str(diff_view.get("candidate_aligned", "")),
                    "marker_line": str(diff_view.get("marker_line", "")),
                    "marker_legend": ".=same, -=realtime extra, +=realtime missing (vs whisperx reference)",
                },
                "html_diff_view": {
                    "compare_unit": str(html_diff_view.get("compare_unit", "char")),
                    "reference_aligned": str(html_diff_view.get("reference_aligned", "")),
                    "realtime_aligned": str(html_diff_view.get("candidate_aligned", "")),
                    "marker_line": str(html_diff_view.get("marker_line", "")),
                },
            }
            # Interactive compare.html: pick any 2 of {srt, spk_subtitles, direct, realtime},
            # toggle 有/沒 spk, S/T-folded diff + metrics table. Missing sources grey out.
            spk_truth_path = Path(input_path).parent / "spk_subtitles"
            spk_truth_text = _try_read_text(str(spk_truth_path)) if spk_truth_path.exists() else ""
            compare_sources = {
                "srt": {"nospk": reference_text or "", "spk": ""},
                "spk_subtitles": {
                    "nospk": _strip_speaker_markers_text(spk_truth_text),
                    "spk": spk_truth_text,
                },
                "direct": {"nospk": direct_nospk_text, "spk": direct_text_for_html},
                "realtime": {"nospk": realtime_nospk_text, "spk": realtime_text_for_html},
            }
            compare_meta_line = (
                f"model={html_lib.escape(str(cfg.stt_model_path or cfg.model_size))} | "
                f"language={html_lib.escape(str(cfg.source_language or 'auto'))} | "
                f"reconcile={float(args.speaker_profile_reconcile_threshold):.2f} | "
                f"match={float(cfg.whisperx_speaker_profile_match_threshold):.2f} | "
                f"min_seconds={float(cfg.whisperx_speaker_profile_min_seconds):.1f} | "
                f"direct↔realtime CER={cer:.4f}"
            )
            compare_html = _build_interactive_compare_html(
                input_path=str(input_path),
                meta_line=compare_meta_line,
                sources=compare_sources,
                language_hint=language_hint,
                spk_vs_truth=spk_vs_truth,
            )
            (case_dir / "compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (case_dir / "compare.txt").write_text(
                "\n".join(
                    [
                        f"input={input_path}",
                        f"model={cfg.stt_model_path or cfg.model_size}",
                        f"language={cfg.source_language or 'auto'}",
                        f"align_model={str(getattr(cfg, 'whisperx_alignment_model', '') or 'auto')}",
                        f"align_language={str(getattr(cfg, 'whisperx_alignment_language', '') or 'auto')}",
                        f"align_device={str(getattr(cfg, 'whisperx_alignment_device', 'auto') or 'auto')}",
                        f"device={cfg.model_device}",
                        f"require_gpu={require_gpu}",
                        f"profile={args.profile}",
                        f"segment_seconds={cfg.segment_seconds}",
                        f"hop_seconds={cfg.hop_seconds}",
                        f"speaker_profile_match_threshold={float(cfg.whisperx_speaker_profile_match_threshold)}",
                        f"speaker_profile_min_seconds={float(cfg.whisperx_speaker_profile_min_seconds)}",
                        f"speaker_profile_reconcile_threshold={float(args.speaker_profile_reconcile_threshold)}",
                        f"speaker_label_max_speakers={int(args.speaker_label_max_speakers)}",
                        f"effective_realtime_speaker_label_max_speakers={int(effective_realtime_speaker_label_max)}",
                        f"direct_requested_chunk_seconds={float(args.direct_chunk_seconds)}",
                        f"direct_chunk_seconds={float(direct_summary.get('direct_chunk_seconds') or 0.0)}",
                        f"direct_language_subchunk_seconds={float(direct_summary.get('direct_language_subchunk_seconds') or 0.0)}",
                        f"direct_chunk_mode={str(direct_summary.get('direct_chunk_mode') or '')}",
                        f"direct_auto_chunked={str(bool(direct_summary.get('direct_auto_chunked', False))).lower()}",
                        f"compare_unit={str(diff_view.get('compare_unit', 'char'))}",
                        f"normalized_char_distance={distance}",
                        f"normalized_cer={cer:.6f}",
                        f"same_units={int(diff_view.get('same_count', 0))}",
                        f"realtime_extra_units={int(diff_view.get('candidate_extra_count', 0))}",
                        f"realtime_missing_units={int(diff_view.get('candidate_missing_count', 0))}",
                        "",
                        "[speaker_compare]",
                        f"enabled={str(bool(speaker_compare.get('enabled', False))).lower()}",
                        f"disabled_reason={speaker_compare.get('disabled_reason', '')}",
                        f"reference_speaker_count={speaker_compare.get('reference_speaker_count', 0)}",
                        f"realtime_speaker_count={speaker_compare.get('realtime_speaker_count', 0)}",
                        f"reference_switch_count={speaker_compare.get('reference_switch_count', 0)}",
                        f"realtime_switch_count={speaker_compare.get('realtime_switch_count', 0)}",
                        f"speaker_sequence_distance={speaker_compare.get('speaker_sequence_distance', 0)}",
                        f"speaker_sequence_error_rate={float(speaker_compare.get('speaker_sequence_error_rate', 0.0)):.6f}",
                        f"realtime_extra_speaker_labels={','.join(speaker_compare.get('realtime_extra_speaker_labels', []))}",
                        f"realtime_missing_speaker_labels={','.join(speaker_compare.get('realtime_missing_speaker_labels', []))}",
                        "",
                        "[speaker_profile_diagnostics.direct]",
                        *_format_speaker_profile_diagnostics(direct_speaker_profile_diagnostics),
                        "",
                        "[speaker_profile_diagnostics.realtime]",
                        *_format_speaker_profile_diagnostics(realtime_speaker_profile_diagnostics),
                        "",
                        "[legend]",
                        ". = same",
                        "- = realtime extra (vs whisperx)",
                        "+ = realtime missing (vs whisperx)",
                        "",
                        "[whisperx_reference_one_line]",
                        dir_norm,
                        "",
                        "[diff_marker_line]",
                        str(diff_view.get("marker_line", "")),
                        "",
                        "[realtime_one_line]",
                        inc_norm,
                        "",
                        "[whisperx_project_txt_plain_one_line]",
                        direct_text_for_compare,
                        "",
                        "[realtime_project_txt_plain_one_line]",
                        realtime_text_for_compare,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (case_dir / "compare.html").write_text(compare_html + "\n", encoding="utf-8")
            (case_dir / "direct_for_compare.txt").write_text(_break_on_speaker_switch(direct_text_for_compare) + "\n", encoding="utf-8")
            (case_dir / "realtime_for_compare.txt").write_text(_break_on_speaker_switch(realtime_text_for_compare) + "\n", encoding="utf-8")
            (case_dir / "compare_diff_marker.txt").write_text(str(diff_view.get("marker_line", "")) + "\n", encoding="utf-8")
            batch_results.append(
                {
                    "input": str(input_path),
                    "case_dir": str(case_dir),
                    "status": "ok",
                    "normalized_cer": float(cer),
                    "realtime_factor": float((incremental_summary.get("timing_summary") or {}).get("realtime_factor", 0.0)),
                    "dominant_stage": str((incremental_summary.get("timing_summary") or {}).get("dominant_stage", "")),
                    "normalized_char_distance": int(distance),
                    "incremental_chars": int(len(inc_norm)),
                    "direct_chars": int(len(dir_norm)),
                    "compare_unit": str(diff_view.get("compare_unit", "char")),
                    "realtime_extra_units": int(diff_view.get("candidate_extra_count", 0)),
                    "realtime_missing_units": int(diff_view.get("candidate_missing_count", 0)),
                    "speaker_compare_enabled": bool(speaker_compare.get("enabled", False)),
                    "reference_speakers": int(speaker_compare.get("reference_speaker_count", 0)),
                    "realtime_speakers": int(speaker_compare.get("realtime_speaker_count", 0)),
                    "speaker_sequence_error_rate": float(speaker_compare.get("speaker_sequence_error_rate", 0.0)),
                    "direct_chunk_seconds": float(direct_summary.get("direct_chunk_seconds") or 0.0),
                    "direct_requested_chunk_seconds": float(direct_summary.get("direct_requested_chunk_seconds") or 0.0),
                    "direct_language_subchunk_seconds": float(
                        direct_summary.get("direct_language_subchunk_seconds") or 0.0
                    ),
                    "direct_chunk_mode": str(direct_summary.get("direct_chunk_mode") or ""),
                    "direct_auto_chunked": bool(direct_summary.get("direct_auto_chunked", False)),
                    **({"vs_reference": vs_reference} if vs_reference else {}),
                    **({"spk_vs_truth": spk_vs_truth} if spk_vs_truth else {}),
                }
            )
            success_rows.append(batch_results[-1])
            try:
                del full_audio
            except Exception:
                pass
            _release_runtime_memory("case-end cleanup")
            print(f"[ok] case report: {case_dir / 'compare.json'}", flush=True)

        avg_cer = sum((float(item["normalized_cer"]) for item in success_rows), 0.0) / float(max(1, len(success_rows)))
        reference_rows = [item for item in success_rows if isinstance(item.get("vs_reference"), dict)]

        def _avg_reference_cer(field: str) -> float | None:
            vals = [
                float(item["vs_reference"][field]["normalized_cer"])
                for item in reference_rows
                if isinstance(item["vs_reference"].get(field), dict)
            ]
            return (sum(vals) / float(len(vals))) if vals else None

        avg_reference_cer_direct = _avg_reference_cer("direct")
        avg_reference_cer_realtime = _avg_reference_cer("realtime")
        speaker_rows = [item for item in success_rows if bool(item.get("speaker_compare_enabled", False))]
        avg_speaker_error = (
            sum(float(item.get("speaker_sequence_error_rate", 0.0)) for item in speaker_rows)
            / float(max(1, len(speaker_rows)))
        )
        spk_truth_rows = [
            item for item in success_rows
            if isinstance(item.get("spk_vs_truth"), dict) and "error" not in item["spk_vs_truth"]
        ]

        def _avg_spk_truth(field: str) -> float | None:
            vals = [float(item["spk_vs_truth"][field]) for item in spk_truth_rows if field in item["spk_vs_truth"]]
            return (sum(vals) / float(len(vals))) if vals else None

        avg_spk_acc_truth_direct = _avg_spk_truth("direct_speaker_accuracy")
        avg_spk_acc_truth_realtime = _avg_spk_truth("realtime_speaker_accuracy")
        decode_knobs = _decode_knob_summary(cfg)
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_root": str(out_root),
            "log_file": str(log_path),
            "case_count": int(len(batch_results)),
            "config": {
                "model": cfg.stt_model_path or cfg.model_size,
                "language": cfg.source_language or "auto",
                "align_model": str(getattr(cfg, "whisperx_alignment_model", "") or "auto"),
                "align_language": str(getattr(cfg, "whisperx_alignment_language", "") or "auto"),
                "align_device": str(getattr(cfg, "whisperx_alignment_device", "auto") or "auto"),
                "device": cfg.model_device,
                **decode_knobs,
                "require_gpu": bool(require_gpu),
                "profile": str(args.profile),
                "segment_seconds": cfg.segment_seconds,
                "hop_seconds": cfg.hop_seconds,
                "speaker_profile_match_threshold": float(cfg.whisperx_speaker_profile_match_threshold),
                "speaker_profile_min_seconds": float(cfg.whisperx_speaker_profile_min_seconds),
                "speaker_profile_reconcile_threshold": float(args.speaker_profile_reconcile_threshold),
                "speaker_label_max_speakers": int(args.speaker_label_max_speakers),
                "direct_group_seconds": float(args.direct_group_seconds),
                "direct_requested_chunk_seconds": float(args.direct_chunk_seconds),
                "direct_language_subchunk_seconds": float(args.direct_language_subchunk_seconds),
                "realtime_compare_one_line": bool(args.realtime_compare_one_line),
                "formats": formats,
                "include_timestamps": include_timestamps,
                "include_speaker": include_speaker,
            },
            "metrics": {
                "avg_normalized_cer": float(avg_cer),
                "max_normalized_cer": float(max((item["normalized_cer"] for item in success_rows), default=0.0)),
                "min_normalized_cer": float(min((item["normalized_cer"] for item in success_rows), default=0.0)),
                "reference_compare_cases": int(len(reference_rows)),
                "avg_reference_cer_direct": float(avg_reference_cer_direct) if avg_reference_cer_direct is not None else None,
                "avg_reference_cer_realtime": float(avg_reference_cer_realtime) if avg_reference_cer_realtime is not None else None,
                "speaker_compare_enabled_cases": int(len(speaker_rows)),
                "avg_speaker_sequence_error_rate": float(avg_speaker_error) if speaker_rows else None,
                "max_speaker_sequence_error_rate": (
                    float(max((float(item.get("speaker_sequence_error_rate", 0.0)) for item in speaker_rows), default=0.0))
                    if speaker_rows
                    else None
                ),
                "spk_truth_cases": int(len(spk_truth_rows)),
                "avg_spk_acc_truth_direct": float(avg_spk_acc_truth_direct) if avg_spk_acc_truth_direct is not None else None,
                "avg_spk_acc_truth_realtime": float(avg_spk_acc_truth_realtime) if avg_spk_acc_truth_realtime is not None else None,
                "failed_cases": int(len(batch_results) - len(success_rows)),
            },
            "cases": batch_results,
        }
        (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_lines = [
            f"output_root={out_root}",
            f"log_file={log_path}",
            f"case_count={len(batch_results)}",
            f"success_count={len(success_rows)}",
            f"model={cfg.stt_model_path or cfg.model_size}",
            f"language={cfg.source_language or 'auto'}",
            f"align_model={str(getattr(cfg, 'whisperx_alignment_model', '') or 'auto')}",
            f"align_language={str(getattr(cfg, 'whisperx_alignment_language', '') or 'auto')}",
            f"align_device={str(getattr(cfg, 'whisperx_alignment_device', 'auto') or 'auto')}",
            f"device={cfg.model_device}",
            f"compute_type={decode_knobs['compute_type']}",
            f"beam_size={decode_knobs['beam_size']}",
            f"batch_size={decode_knobs['batch_size']}",
            f"require_gpu={require_gpu}",
            f"profile={args.profile}",
            f"segment_seconds={cfg.segment_seconds}",
            f"hop_seconds={cfg.hop_seconds}",
            f"speaker_profile_match_threshold={float(cfg.whisperx_speaker_profile_match_threshold)}",
            f"speaker_profile_min_seconds={float(cfg.whisperx_speaker_profile_min_seconds)}",
            f"speaker_profile_reconcile_threshold={float(args.speaker_profile_reconcile_threshold)}",
            f"speaker_label_max_speakers={int(args.speaker_label_max_speakers)}",
            f"direct_group_seconds={float(args.direct_group_seconds)}",
            f"direct_requested_chunk_seconds={float(args.direct_chunk_seconds)}",
            f"direct_language_subchunk_seconds={float(args.direct_language_subchunk_seconds)}",
            f"realtime_compare_one_line={bool(args.realtime_compare_one_line)}",
            f"avg_normalized_cer={avg_cer:.6f}",
            f"reference_compare_cases={len(reference_rows)}",
            f"avg_reference_cer_direct={avg_reference_cer_direct:.6f}"
            if avg_reference_cer_direct is not None
            else "avg_reference_cer_direct=none",
            f"avg_reference_cer_realtime={avg_reference_cer_realtime:.6f}"
            if avg_reference_cer_realtime is not None
            else "avg_reference_cer_realtime=none",
            f"speaker_compare_enabled_cases={len(speaker_rows)}",
            f"avg_speaker_sequence_error_rate={avg_speaker_error:.6f}"
            if speaker_rows
            else "avg_speaker_sequence_error_rate=disabled",
            f"spk_truth_cases={len(spk_truth_rows)}",
            f"avg_spk_acc_truth_direct={avg_spk_acc_truth_direct:.4f}"
            if avg_spk_acc_truth_direct is not None
            else "avg_spk_acc_truth_direct=none",
            f"avg_spk_acc_truth_realtime={avg_spk_acc_truth_realtime:.4f}"
            if avg_spk_acc_truth_realtime is not None
            else "avg_spk_acc_truth_realtime=none",
            "",
            "cases:",
        ]
        for item in batch_results:
            if str(item.get("status", "")) == "ok":
                summary_lines.append(
                    f"- input={item['input']} | cer={float(item['normalized_cer']):.6f} | "
                    f"rtf={float(item.get('realtime_factor', 0.0)):.3f}x | "
                    f"unit={str(item.get('compare_unit', 'char'))} | "
                    f"extra={int(item.get('realtime_extra_units', 0))} | "
                    f"missing={int(item.get('realtime_missing_units', 0))} | "
                    + (
                        f"spk_ref={int(item.get('reference_speakers', 0))} | "
                        f"spk_rt={int(item.get('realtime_speakers', 0))} | "
                        f"spk_err={float(item.get('speaker_sequence_error_rate', 0.0)):.6f} | "
                        if bool(item.get("speaker_compare_enabled", False))
                        else "speaker_compare=disabled | "
                    )
                    + (
                        (
                            f"ref_cer_direct={float(item['vs_reference']['direct']['normalized_cer']):.6f} | "
                            + (
                                f"ref_cer_realtime={float(item['vs_reference']['realtime']['normalized_cer']):.6f} | "
                                if isinstance(item["vs_reference"].get("realtime"), dict)
                                else ""
                            )
                        )
                        if isinstance(item.get("vs_reference"), dict) and isinstance(item["vs_reference"].get("direct"), dict)
                        else ""
                    )
                    + (
                        (
                            f"spk_acc_truth_direct={float(item['spk_vs_truth'].get('direct_speaker_accuracy', 0.0)):.4f} | "
                            f"spk_acc_truth_rt={float(item['spk_vs_truth'].get('realtime_speaker_accuracy', 0.0)):.4f} | "
                            f"spk_truth_ref={int(item['spk_vs_truth'].get('ref_speakers', 0))} | "
                            f"spk_truth_pred_direct={int(item['spk_vs_truth'].get('direct_pred_speakers', 0))} | "
                            f"spk_truth_pred_rt={int(item['spk_vs_truth'].get('realtime_pred_speakers', 0))} | "
                        )
                        if isinstance(item.get("spk_vs_truth"), dict) and "error" not in item["spk_vs_truth"]
                        else ""
                    )
                    + f"case_dir={item['case_dir']}"
                )
            else:
                summary_lines.append(
                    f"- input={item['input']} | status=failed | error={item.get('error', '')} | case_dir={item['case_dir']}"
                )
        (out_root / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"[ok] summary json: {out_root / 'summary.json'}", flush=True)
        print(f"[ok] summary txt : {out_root / 'summary.txt'}", flush=True)
        print(f"[ok] run log    : {log_path}", flush=True)
        return 0
    finally:
        restore_logger()


if __name__ == "__main__":
    raise SystemExit(main())
