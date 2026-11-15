"""Centralized, privacy-safe diagnostics for Pi agent runs."""
from __future__ import annotations
import functools
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

def configure_logging(level="INFO", log_file=None):
    """Write progress to stderr and a 10 MiB rotating log file when requested."""
    level = str(level).upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError("Invalid log level")
    logger = logging.getLogger("harness")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False
    for h in list(logger.handlers):
        if getattr(h, "_a1_handler", False):
            logger.removeHandler(h)
            h.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream._a1_handler = True
    logger.addHandler(stream)
    dest = log_file if log_file is not None else os.environ.get("A1_LOG_FILE")
    if dest and str(dest) != "-":
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(p, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
        h.setFormatter(formatter)
        h._a1_handler = True
        logger.addHandler(h)
    logger.info("logging_ready level=%s file_enabled=%s", level, bool(dest and str(dest) != "-"))
    return logger

def trace(fn):
    """Opt-in DEBUG function enter/exit, duration, and exception type. Never arguments."""
    log = logging.getLogger("harness.trace")
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not log.isEnabledFor(logging.DEBUG):
            return fn(*args, **kwargs)
        start = time.monotonic()
        log.debug("enter fn=%s", fn.__qualname__)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            log.error("failed fn=%s type=%s elapsed_ms=%.1f",
                      fn.__qualname__, type(exc).__name__, (time.monotonic()-start)*1000)
            raise
        log.debug("exit fn=%s elapsed_ms=%.1f", fn.__qualname__, (time.monotonic()-start)*1000)
        return result
    return wrapper
