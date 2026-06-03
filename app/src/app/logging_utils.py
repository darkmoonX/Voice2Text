"""Logger setup utilities used by bootstrap and controller modules."""
from __future__ import annotations
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


_ROOT_FILE_HANDLER_MARKER = "_voice2text_root_file_handler"
_SUPPRESSED_CONSOLE_HANDLER_MARKER = "_voice2text_suppressed_console_handler"


def configure_app_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger('voice2text')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / 'voice2text.log'
    target_path = str(log_file.resolve())
    for handler in logger.handlers:
        base_name = getattr(handler, 'baseFilename', None)
        if isinstance(base_name, str) and base_name == target_path:
            configure_third_party_file_logging(log_dir)
            return logger
    formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = TimedRotatingFileHandler(filename=target_path, when='midnight', backupCount=7, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    configure_third_party_file_logging(log_dir)
    return logger


def configure_third_party_file_logging(log_dir: str) -> None:
    """Capture library loggers to file without letting warning fallback spam stderr."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    target_path = str((path / "voice2text.log").resolve())
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in root.handlers:
        if bool(getattr(handler, _ROOT_FILE_HANDLER_MARKER, False)):
            base_name = getattr(handler, "baseFilename", None)
            if isinstance(base_name, str) and base_name == target_path:
                return
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = TimedRotatingFileHandler(
        filename=target_path,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    setattr(file_handler, _ROOT_FILE_HANDLER_MARKER, True)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.WARNING)
    root.addHandler(file_handler)
    suppress_third_party_console_logging()


def suppress_third_party_console_logging() -> None:
    """Remove existing console stream handlers so library warnings stay in app log/debug."""
    loggers: list[logging.Logger] = [logging.getLogger()]
    for logger_obj in logging.Logger.manager.loggerDict.values():
        if isinstance(logger_obj, logging.Logger):
            loggers.append(logger_obj)
    for logger in loggers:
        for handler in list(logger.handlers):
            if bool(getattr(handler, _ROOT_FILE_HANDLER_MARKER, False)):
                continue
            if isinstance(handler, logging.FileHandler):
                continue
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
                setattr(handler, _SUPPRESSED_CONSOLE_HANDLER_MARKER, True)
                try:
                    handler.close()
                except Exception:
                    pass
