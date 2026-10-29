
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level=logging.INFO, only_logger=None):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    if only_logger is not None:
        for handler in logging.getLogger().handlers:
            handler.addFilter(logging.Filter(only_logger))
