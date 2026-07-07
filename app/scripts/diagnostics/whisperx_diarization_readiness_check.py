"""Foreground diagnostics for WhisperX diarization readiness on current machine."""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _print_line(message: str) -> None:
    print(message, flush=True)


def _mask(token: str | None) -> str:
    raw = (token or "").strip()
    if not raw:
        return "(empty)"
    if len(raw) <= 8:
        return "*" * len(raw)
    return raw[:4] + "*" * (len(raw) - 8) + raw[-4:]


def _resolve_token(arg_token: str) -> str | None:
    if arg_token.strip():
        return arg_token.strip()
    settings_file = SRC_ROOT / "runtime_settings.json"
    if settings_file.exists():
        try:
            payload = json.loads(settings_file.read_text(encoding="utf-8"))
            token = str(payload.get("whisperx_hf_token") or "").strip()
            if token:
                return token
        except Exception:
            pass
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return value
    return None


def _check_hf_repo(repo_id: str, token: str | None) -> tuple[bool, str]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        return (False, f"huggingface_hub import failed: {exc}")
    try:
        info = HfApi().model_info(repo_id=repo_id, token=token)
        sha = str(getattr(info, "sha", "") or "")[:10]
        return (True, f"ok sha={sha or 'n/a'}")
    except Exception as exc:  # noqa: BLE001
        return (False, f"{type(exc).__name__}: {exc}")


def _check_pipeline_import() -> tuple[bool, str]:
    try:
        import whisperx  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (False, f"whisperx import failed: {exc}")
    direct_cls = getattr(whisperx, "DiarizationPipeline", None)
    if callable(direct_cls):
        return (True, "whisperx.DiarizationPipeline available")
    try:
        from whisperx.diarize import DiarizationPipeline as module_cls  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (False, f"DiarizationPipeline unavailable: {exc}")
    return (True, f"whisperx.diarize.DiarizationPipeline available ({module_cls})")


def _check_proxy_env() -> list[tuple[str, str]]:
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    )
    rows: list[tuple[str, str]] = []
    for key in keys:
        val = str(os.environ.get(key, "") or "").strip()
        if val:
            rows.append((key, val))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="WhisperX diarization readiness checker.")
    parser.add_argument("--hf-token", default="", help="HF token override.")
    parser.add_argument(
        "--repos",
        default="pyannote/speaker-diarization-3.1,pyannote/segmentation-3.0,pyannote/wespeaker-voxceleb-resnet34-lm",
        help="Comma-separated HF repos to check.",
    )
    args = parser.parse_args()

    _print_line("=== WhisperX Diarization Readiness ===")
    proxy_rows = _check_proxy_env()
    if proxy_rows:
        _print_line("[env] proxy variables:")
        for (key, value) in proxy_rows:
            _print_line(f"  - {key}={value}")
    else:
        _print_line("[env] no proxy variables detected")

    token = _resolve_token(args.hf_token)
    _print_line(f"[env] hf_token={_mask(token)}")

    (pipeline_ok, pipeline_detail) = _check_pipeline_import()
    _print_line(f"[check] pipeline_import={'OK' if pipeline_ok else 'FAIL'}: {pipeline_detail}")

    repo_ids = [item.strip() for item in args.repos.split(",") if item.strip()]
    all_ok = pipeline_ok
    for repo_id in repo_ids:
        (ok, detail) = _check_hf_repo(repo_id, token)
        _print_line(f"[check] repo={repo_id} {'OK' if ok else 'FAIL'}: {detail}")
        all_ok = all_ok and ok

    if all_ok:
        _print_line("[result] READY: diarization dependencies are reachable from this environment.")
        return 0
    _print_line("[result] NOT READY: fix proxy/token/network first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
