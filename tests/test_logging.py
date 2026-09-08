import io
import logging

from financial_agent.logging_config import configure_logging


def test_reconfiguration_does_not_duplicate_logs_or_modify_root():
    root_handlers = list(logging.getLogger().handlers)
    root_level = logging.getLogger().level
    stream = io.StringIO()
    logger = configure_logging("DEBUG", stream=stream)
    try:
        configure_logging("INFO", stream=stream)
        logger.debug("hidden-debug")
        logger.info("bootstrap-ready")
        output = stream.getvalue()
        assert output.count("bootstrap-ready") == 1
        assert "hidden-debug" not in output
        assert "INFO financial_agent" in output
        assert logging.getLogger().handlers == root_handlers
        assert logging.getLogger().level == root_level
    finally:
        for handler in list(logger.handlers):
            if getattr(handler, "_financial_agent_owned", False):
                logger.removeHandler(handler)
                handler.close()
