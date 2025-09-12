__version__ = "0.1.0"

import logging

from ppcluster.utils.logger import get_logger, set_log_level, setup_logger  # noqa: F401

logger = setup_logger(logging.INFO, name="ppcx", force=True)
