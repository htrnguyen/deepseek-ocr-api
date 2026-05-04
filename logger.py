import logging
from datetime import datetime, timezone, timedelta


def custom_time(*args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    tz_dt = utc_dt.astimezone(timezone(timedelta(hours=7)))
    return tz_dt.timetuple()


def setup_logger() -> logging.Logger:
    logging.Formatter.converter = custom_time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s +0700 | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("deepseek-ocr-api")


logger = setup_logger()
