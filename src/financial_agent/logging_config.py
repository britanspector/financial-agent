"""Configure only the project logger; never log request bodies or settings."""

import logging
from typing import TextIO


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> logging.Logger:
    logger = logging.getLogger("financial_agent")
    logger.setLevel(level)
    # Own only handlers created here; repeated configuration must not duplicate output.
    for handler in list(logger.handlers):
        if getattr(handler, "_financial_agent_owned", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream)
    handler._financial_agent_owned = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
