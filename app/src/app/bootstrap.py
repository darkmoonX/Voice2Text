"""Application entry bootstrap: parse args, build config, run checks/runtime."""
from __future__ import annotations

from pathlib import Path

from .bootstrap_args import build_arg_parser, print_app_sessions, print_devices
from .bootstrap_config import build_runtime_config, default_log_dir
from .bootstrap_runtime import run_qt_app
from .settings.presets import apply_preset_defaults
from .stt import has_failed_reports, run_provider_health_check, summarize_health_reports
from .whisper_config import load_whisper_runtime_params


def main(argv: list[str] | None = None) -> int:
    config_path = Path(__file__).resolve().parents[1] / "whisper_config.json"
    whisper_defaults = load_whisper_runtime_params(config_path)
    parser = build_arg_parser(whisper_defaults)
    parser.set_defaults(log_dir=default_log_dir())
    apply_preset_defaults(parser, argv)
    args = parser.parse_args(argv)
    if args.list_devices:
        return print_devices()
    if args.list_app_sessions:
        return print_app_sessions()
    cfg = build_runtime_config(args)
    if args.stt_health_check:
        reports = run_provider_health_check(cfg, scope=args.stt_health_check_scope)
        print(summarize_health_reports(reports))
        return 2 if has_failed_reports(reports) else 0
    return run_qt_app(cfg)
