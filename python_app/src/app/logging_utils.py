"""Logger setup utilities used by bootstrap and controller modules."""
from __future__ import annotations
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

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
            return logger
    formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = TimedRotatingFileHandler(filename=target_path, when='midnight', backupCount=7, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger
