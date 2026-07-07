from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest
from io import StringIO

from voice2text.logging_utils import WindowsSafeTimedRotatingFileHandler, configure_app_logger


class _RolloverPermissionErrorHandler(WindowsSafeTimedRotatingFileHandler):
    def doRollover(self) -> None:
        raise PermissionError(32, "file is locked")


class LoggingUtilsTests(unittest.TestCase):
    def test_third_party_warnings_go_to_file_without_console_handler(self) -> None:
        root = logging.getLogger()
        app_logger = logging.getLogger("voice2text")
        original_handlers = list(root.handlers)
        original_app_handlers = list(app_logger.handlers)
        try:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in list(app_logger.handlers):
                app_logger.removeHandler(handler)
            root.addHandler(logging.StreamHandler(StringIO()))
            with tempfile.TemporaryDirectory(prefix="v2t-log-") as td:
                log_dir = Path(td)
                configure_app_logger(str(log_dir))
                logging.getLogger("whisperx.vads.pyannote").warning("No active speech found in audio")
                for handler in root.handlers:
                    handler.flush()
                text = (log_dir / "voice2text.log").read_text(encoding="utf-8")
                self.assertIn("whisperx.vads.pyannote", text)
                self.assertIn("No active speech found in audio", text)
                stream_handlers = [
                    handler
                    for handler in root.handlers
                    if isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)
                ]
                self.assertEqual(stream_handlers, [])
                for handler in list(root.handlers):
                    root.removeHandler(handler)
                    handler.close()
                for handler in list(app_logger.handlers):
                    app_logger.removeHandler(handler)
                    handler.close()
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
                handler.close()
            for handler in list(app_logger.handlers):
                app_logger.removeHandler(handler)
                handler.close()
            for handler in original_handlers:
                root.addHandler(handler)
            for handler in original_app_handlers:
                app_logger.addHandler(handler)

    def test_locked_rollover_switches_to_fallback_log_without_console_error(self) -> None:
        logger = logging.getLogger("voice2text.test.locked-rollover")
        original_handlers = list(logger.handlers)
        original_propagate = logger.propagate
        original_raise_exceptions = logging.raiseExceptions
        try:
            logging.raiseExceptions = True
            logger.handlers.clear()
            logger.propagate = False
            with tempfile.TemporaryDirectory(prefix="v2t-log-lock-") as td:
                log_dir = Path(td)
                handler = _RolloverPermissionErrorHandler(
                    filename=str(log_dir / "voice2text.log"),
                    when="midnight",
                    backupCount=7,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                handler.rolloverAt = 0
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)

                logger.info("message after locked rollover")
                handler.flush()
                logger.removeHandler(handler)
                handler.close()

                fallback_logs = list(log_dir.glob("voice2text.*.pid*.log"))
                self.assertEqual(len(fallback_logs), 1)
                self.assertIn("message after locked rollover", fallback_logs[0].read_text(encoding="utf-8"))
        finally:
            logging.raiseExceptions = original_raise_exceptions
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            for handler in original_handlers:
                logger.addHandler(handler)
            logger.propagate = original_propagate


if __name__ == "__main__":
    unittest.main()
