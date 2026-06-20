"""Logger setup utilities used by bootstrap and controller modules."""
from __future__ import annotations
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import sys
import time


_ROOT_FILE_HANDLER_MARKER = "_voice2text_root_file_handler"
_SUPPRESSED_CONSOLE_HANDLER_MARKER = "_voice2text_suppressed_console_handler"


class WindowsSafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Timed rotating handler that survives Windows file-lock rollover failures."""

    def handleError(self, record: logging.LogRecord) -> None:
        exc_type, exc, _tb = sys.exc_info()
        if isinstance(exc, PermissionError) or (
            isinstance(exc, OSError) and getattr(exc, "winerror", None) == 32
        ):
            if self._switch_to_fallback_file():
                try:
                    logging.FileHandler.emit(self, record)
                except Exception:
                    pass
            return
        super().handleError(record)

    def _switch_to_fallback_file(self) -> bool:
        original = Path(self.baseFilename)
        fallback = original.with_name(
            f"{original.stem}.{time.strftime('%Y-%m-%d')}.pid{os.getpid()}{original.suffix}"
        )
        try:
            if self.stream:
                self.stream.close()
                self.stream = None
            self.baseFilename = str(fallback)
            self.stream = self._open()
            current_time = int(time.time())
            self.rolloverAt = self.computeRollover(current_time)
            return True
        except Exception:
            return False


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
    file_handler = WindowsSafeTimedRotatingFileHandler(filename=target_path, when='midnight', backupCount=7, encoding='utf-8')
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
    file_handler = WindowsSafeTimedRotatingFileHandler(
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
