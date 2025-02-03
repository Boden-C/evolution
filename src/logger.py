"""
Enterprise-level logging configuration with color console output and structured file logging.
"""
from __future__ import annotations

import logging
import os
import sys
from logging import Logger
from logging.handlers import RotatingFileHandler

try:
    import colorlog
except ImportError:
    colorlog = None

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    jsonlogger = None


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
        # Avoid adding handlers multiple times
        if logger.handlers:
            return logger

        # Set overall log level
        logger.setLevel(cls.DEFAULT_LOG_LEVEL)

        # Console handler
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

        # Ensure log directory exists
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        file_path = os.path.join(cls.LOG_DIR, cls.LOG_FILE)

        # Rotating file handler
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
