from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover - fallback path
    _loguru_logger = None

_print_level = "INFO"
_initialized = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\x1b[36m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[35m",
    }
    RESET = "\x1b[0m"
    BASE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelno, "")
        if color:
            record.levelname = f"{color}{original_levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _parse_level(level_name: str, default_level: int) -> int:
    level = getattr(logging, level_name.strip().upper(), None)
    return level if isinstance(level, int) else default_level


def _configure_std_logging(print_level: str, logfile_level: str, logfile_path: Path) -> None:
    package_logger = logging.getLogger("agent_core")
    package_logger.handlers.clear()
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(_parse_level(print_level, logging.INFO))
    stream_handler.setFormatter(_ColorFormatter(_ColorFormatter.BASE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    file_handler = logging.FileHandler(logfile_path, encoding="utf-8")
    file_handler.setLevel(_parse_level(logfile_level, logging.DEBUG))
    file_handler.setFormatter(logging.Formatter(_ColorFormatter.BASE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    package_logger.addHandler(stream_handler)
    package_logger.addHandler(file_handler)


def define_log_level(
    print_level: str = "INFO",
    logfile_level: str = "DEBUG",
    name: Optional[str] = None,
):
    global _print_level, _initialized
    _print_level = print_level

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{timestamp}" if name else timestamp

    log_dir = PROJECT_ROOT / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = log_dir / f"{log_name}.log"

    if _loguru_logger is not None:
        _loguru_logger.remove()

        # Keep loguru's default style to match previous project appearance.
        _loguru_logger.add(sys.stderr, level=print_level.strip().upper())
        _loguru_logger.add(logfile_path, level=logfile_level.strip().upper())
    else:
        _configure_std_logging(print_level, logfile_level, logfile_path)

    _initialized = True
    return get_logger("agent_core")


def get_logger(
    name: str,
    print_level: str = "INFO",
    logfile_level: str = "DEBUG",
    log_name: Optional[str] = None,
):
    """Get logger instance with lazy initialization."""
    global _initialized
    if not _initialized:
        define_log_level(
            print_level=os.getenv("LOG_LEVEL", print_level),
            logfile_level=os.getenv("LOG_FILE_LEVEL", logfile_level),
            name=os.getenv("LOG_NAME", log_name),
        )

    if _loguru_logger is not None:
        return _loguru_logger.bind(module=name)
    return logging.getLogger(name)


# Default initialization for backward compatibility.
logger = get_logger("agent_core")
