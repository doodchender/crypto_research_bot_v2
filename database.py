import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pytz

MOSCOW_TZ = pytz.timezone("Europe/Moscow")
DB_RETENTION_DAYS = 30

def get_db_path(cfg: dict) -> str:
    data_dir = Path(cfg["paths"]["data"])
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "hourly.db")

def init_db(cfg: dict, logger: logging.Logger):
    db_path = get_db_path(cfg)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_snapshots (
                timestamp           TEXT PRIMARY KEY,
                btc_price           REAL,
                eth_price           REAL,
                btc_atr_pct         REAL,
                eth_atr_pct         REAL,
                fear_greed_value    INTEGER,
                fear_greed_label    TEXT,
                media_sentiment     REAL,
                media_articles_count INTEGER,
                btc_oi_usd          REAL,
                eth_oi_usd          REAL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"DB инициализирована: {db_path}")
    except Exception as e:
        logger.error(f"DB init error: {e}")

def save_snapshot(cfg: dict, logger: logging.Logger, data: dict):
    db_path = get_db_path(cfg)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR REPLACE INTO hourly_snapshots (
                timestamp, btc_price, eth_price,
                btc_atr_pct, eth_atr_pct,
                fear_greed_value, fear_greed_label,
                media_sentiment, media_articles_count,
                btc_oi_usd, eth_oi_usd
            ) VALUES (
                :timestamp, :btc_price, :eth_price,
                :btc_atr_pct, :eth_atr_pct,
                :fear_greed_value, :fear_greed_label,
                :media_sentiment, :media_articles_count,
                :btc_oi_usd, :eth_oi_usd
            )
        """, data)
        conn.commit()
        conn.close()
        logger.info(f"DB snapshot сохранён: {data['timestamp']}")
    except Exception as e:
        logger.error(f"DB save error: {e}")

def get_today_snapshots(cfg: dict, logger: logging.Logger) -> list:
    db_path = get_db_path(cfg)
    today = datetime.now(MOSCOW_TZ).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM hourly_snapshots WHERE timestamp LIKE ? ORDER BY timestamp",
            (f"{today}%",)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return []

def get_snapshots_range(cfg: dict, logger: logging.Logger,
                        start: str, end: str) -> list:
    db_path = get_db_path(cfg)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM hourly_snapshots WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start + " 00:00", end + " 23:59")
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"DB range read error: {e}")
        return []

def cleanup_old_records(cfg: dict, logger: logging.Logger):
    db_path = get_db_path(cfg)
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(days=DB_RETENTION_DAYS)).strftime("%Y-%m-%d %H:%M")
    try:
        conn = sqlite3.connect(db_path)
        deleted = conn.execute(
            "DELETE FROM hourly_snapshots WHERE timestamp < ?", (cutoff,)
        ).rowcount
        conn.commit()
        conn.close()
        if deleted:
            logger.info(f"DB cleanup: удалено {deleted} старых записей")
    except Exception as e:
        logger.error(f"DB cleanup error: {e}")
