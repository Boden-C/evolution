"""
Enterprise-grade logging utilities.

Provides:
- Color console and rotating structured file logs.
- Distributed-aware log record extras (hostname/ranks).
- Rank-based filtering (global/local rank 0 only or all ranks).
- Robust exception hook for CLI entrypoints.
- Warning suppression and minimal env var hardening for tokenizers.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import socket
import enum
from typing import Optional

try:
    import colorlog
except ImportError:
    colorlog = None

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    jsonlogger = None

try:
    from rich.console import Console  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Console = None  # type: ignore

try:
    from .exceptions import CliError
except Exception:  # pragma: no cover - avoid hard import failure at import time
    class CliError(Exception):  # type: ignore
        pass


# Extra fields injected into LogRecord
_log_extra_fields: dict[str, object] = {}


class LogFilterType(str, enum.Enum):
    """Rank-based logging filter strategy."""

    rank0_only = "rank0_only"
    local_rank0_only = "local_rank0_only"
    all_ranks = "all_ranks"


def log_extra_field(name: str, value: object) -> None:
    """Set or clear an extra field to be injected into all LogRecords."""
    if value is None:
        _log_extra_fields.pop(name, None)
    else:
        _log_extra_fields[name] = value


class _RankFilter(logging.Filter):
    """Filter that suppresses INFO-and-below from non-zero ranks."""

    def __init__(self, mode: LogFilterType) -> None:
        super().__init__(name="elevation.rank_filter")
        self.mode = mode

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO:
            return True
        if self.mode == LogFilterType.rank0_only:
            return getattr(record, "global_rank", 0) == 0
        if self.mode == LogFilterType.local_rank0_only:
            return getattr(record, "local_rank", 0) == 0
        return True


_record_factory_installed = False


class Logger:
    DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))
    BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    CONSOLE_COLORS: dict[str, str] = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Create or retrieve a configured logger.
        Console output is colored; file output is rotated and structured (JSON if available).
        """
        logger = logging.getLogger(name)
        _ensure_record_factory()
        if logger.handlers:
            return logger

        logger.setLevel(cls.DEFAULT_LOG_LEVEL)
        logger.propagate = False

        console_handler = logging.StreamHandler(sys.stdout)
        if colorlog:
            fmt = "%(log_color)s" + cls.FORMAT
            console_formatter = colorlog.ColoredFormatter(
                fmt,
                datefmt=cls.DATEFMT,
                log_colors=cls.CONSOLE_COLORS,
            )
        else:
            console_formatter = logging.Formatter(cls.FORMAT, datefmt=cls.DATEFMT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        os.makedirs(cls.LOG_DIR, exist_ok=True)
        file_path = os.path.join(cls.LOG_DIR, cls.LOG_FILE)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=cls.MAX_BYTES,
            backupCount=cls.BACKUP_COUNT,
            encoding="utf-8",
        )
        if jsonlogger:
            file_formatter = jsonlogger.JsonFormatter(
                fmt=cls.FORMAT,
                datefmt=cls.DATEFMT,
            )
        else:
            file_formatter = logging.Formatter(cls.FORMAT, datefmt=cls.DATEFMT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger


# Convenience function
get_logger = Logger.get_logger


def _ensure_record_factory() -> None:
    """Install a LogRecordFactory that injects extra fields, once."""
    global _record_factory_installed
    if _record_factory_installed:
        return
    orig_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        rec = orig_factory(*args, **kwargs)
        for k, v in _log_extra_fields.items():
            try:
                setattr(rec, k, v)
            except Exception:
                pass
        return rec

    logging.setLogRecordFactory(factory)
    _record_factory_installed = True


def setup_logging(filter_type: LogFilterType = LogFilterType.rank0_only) -> None:
    """Augment logging with extras and install a rank-based root filter."""
    # Extras
    log_extra_field("hostname", socket.gethostname())

    # Lazy import to avoid circulars
    try:
        from .util.runtime_management import (
            is_distributed,
            get_node_rank,
            get_local_rank,
            get_global_rank,
        )
    except Exception:
        is_distributed = lambda: False  # type: ignore
        get_node_rank = lambda: 0  # type: ignore
        get_local_rank = lambda: 0  # type: ignore
        get_global_rank = lambda: 0  # type: ignore

    if is_distributed():
        log_extra_field("node_rank", get_node_rank())
        log_extra_field("local_rank", get_local_rank())
        log_extra_field("global_rank", get_global_rank())
    else:
        log_extra_field("node_rank", 0)
        log_extra_field("local_rank", 0)
        log_extra_field("global_rank", 0)

    _ensure_record_factory()

    root_logger = logging.getLogger()

    # Replace existing filter
    current = getattr(root_logger, "_elevation_log_filter", None)
    if current is not None:
        try:
            root_logger.removeFilter(current)
        except Exception:
            pass

    new_filter: Optional[logging.Filter]
    if filter_type in (LogFilterType.rank0_only, LogFilterType.local_rank0_only):
        new_filter = _RankFilter(filter_type)
    elif filter_type == LogFilterType.all_ranks:
        new_filter = None
    else:  # pragma: no cover
        raise ValueError(str(filter_type))

    if new_filter is not None:
        root_logger.addFilter(new_filter)
    setattr(root_logger, "_elevation_log_filter", new_filter)

    # Capture warnings and quiet noisy libs
    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def excepthook(exc_type: type, value: BaseException, tb: object) -> None:
    """Handle uncaught exceptions for CLI UX."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, value, tb)
        return
    if issubclass(exc_type, CliError):
        if Console:
            Console().print(f"[yellow]{value}[/]")
        else:
            sys.stderr.write(f"WARNING: {value}\n")
        return
    logging.critical("Unhandled %s: %s", exc_type.__name__, value, exc_info=(exc_type, value, tb))


def install_excepthook() -> None:
    """Install the CLI-friendly excepthook."""
    sys.excepthook = excepthook


def filter_warnings() -> None:
    """Silence known noisy warnings from third-party libs."""
    import warnings

    patterns = [
        r"torch\.distributed.*",
        r"TypedStorage is deprecated.*",
        r"failed to load.*",
    ]
    for pat in patterns:
        warnings.filterwarnings("ignore", message=pat)


def set_env_vars() -> None:
    """Set safe defaults for environment variables."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def prepare_cli_env(filter_type: Optional[LogFilterType] = None) -> None:
    """Turnkey setup for CLI tools: logging, hooks, warnings, env vars."""
    if filter_type is None:
        val = os.environ.get("LOG_FILTER_TYPE", LogFilterType.rank0_only.value)
        try:
            filter_type = LogFilterType(val) if not isinstance(val, LogFilterType) else val
        except Exception:
            filter_type = LogFilterType.rank0_only
    setup_logging(filter_type)
    install_excepthook()
    filter_warnings()
    set_env_vars()
