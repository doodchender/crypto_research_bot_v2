import time
import logging
import yaml
import requests
from pathlib import Path

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logger(name, cfg):
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_cfg.get("log_to_file", False):
        log_dir = Path(cfg["paths"]["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def get_with_retry(url, params, cfg, logger, headers=None):
    max_tries = cfg["api"].get("max_retries", 3)
    delay     = cfg["api"].get("retry_delay_seconds", 5)
    timeout   = cfg["api"].get("request_timeout", 30)
    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.get(
                url, params=params,
                headers=headers or {}, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            wait = delay * (2 ** (attempt - 1))
            logger.warning(f"HTTP {status}, попытка {attempt}/{max_tries}, жду {wait}с...")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса попытка {attempt}: {e}")
            time.sleep(delay)
    logger.error(f"Все {max_tries} попыток провалились: {url}")
    return None

def ensure_dirs(cfg):
    Path(cfg["paths"]["data"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)

SYMBOL_MAP = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
