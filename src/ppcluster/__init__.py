__version__ = "0.1.0"

import logging

# Import modules
from ppcluster import (
    mcmc,  # noqa: F401
    utils,  # noqa: F401
)

# Import specific functions
from ppcluster.utils.logger import get_logger, set_log_level, setup_logger  # noqa: F401

logger = setup_logger(logging.INFO, name="ppcx", force=True)
