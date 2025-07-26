#!/usr/bin/env python
import logging
import sys

# Get the root logger for the package
logger = logging.getLogger("scfeatureprofiler")

def setup_logging(level=logging.INFO):
    """
    Configures the package's logger.
    """
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)