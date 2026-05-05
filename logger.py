import logging
import sys
from datetime import datetime, timezone, timedelta

_APP_LOGGER_NAME = "ocr-api"


def _utc7_converter(*args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(timezone(timedelta(hours=7))).timetuple()


def setup_logger() -> logging.Logger:
    log = logging.getLogger(_APP_LOGGER_NAME)

    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    log.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt.converter = _utc7_converter

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.addHandler(handler)

    return log


logger = setup_logger()
