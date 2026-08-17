import structlog
import logging
import sys
from app.core.config import settings

def setup_logging():
    if settings.DEBUG:
        processors = [
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer()
        ]
    else:
        processors = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.LOG_LEVEL.upper(),
    )
