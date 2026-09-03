import logging
from pathlib import Path

LOG_PATH = Path(__file__).with_name("market_intelligence.log")

def get_logger(name="market_intelligence"):
    logger=logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh=logging.FileHandler(LOG_PATH,encoding="utf-8")
        fh.setFormatter(fmt)
        sh=logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
