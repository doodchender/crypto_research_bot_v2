import sqlite3
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import pytz

MOSCOW_TZ  = pytz.timezone("Europe/Moscow")
DATA_DIR   = Path("data")
DB_PATH    = DATA_DIR / "hourly.db"
DAYS_BACK  = 30

def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_snapshots (
            timestamp            TEXT PRIMARY KEY,
            btc_price            REAL,
            eth_price            REAL,
            btc_atr_pct          REAL,
            eth_atr_pct          REAL,
            fear_greed_value     INTEGER,
            fear_greed_label     TEXT,
            media_sentiment      REAL,
            media_articles_count INTEGER,
            btc_oi_usd           REAL,
            eth_oi_usd           REAL
        )
    """)
    conn.commit()
    conn.close()
    print("БД инициализирована")

def fetch_prices_and_atr(symbol: str, yf_ticker: str) -> pd.DataFrame:
    print(f"Загружаю {symbol} (yfinance, 30д почасово)...")
    raw = yf.download(yf_ticker, period="30d", interval="1h",
                      progress=False, auto_adjust=True)
    if raw.empty:
        print(f"  {symbol}: нет данных!")
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(MOSCOW_TZ)

    close = raw["Close"]
    high  = raw["High"]
    low   = raw["Low"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr     = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    atr_pct = (atr / close * 100).round(3)

    result = pd.DataFrame({
        f"{symbol.lower()}_price":   close.round(2),
        f"{symbol.lower()}_atr_pct": atr_pct,
    })
    print(f"  {symbol}: {len(result)} часовых точек")
    return result

def fetch_fear_greed(days: int = 30) -> pd.DataFrame:
    print("Загружаю Fear & Greed (alternative.me)...")
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": days * 24, "format": "json"},
            timeout=15
        )
        data = resp.json().get("data", [])
        rows = []
        for item in data:
            ts = datetime.fromtimestamp(int(item["timestamp"]), tz=pytz.UTC)
            ts = ts.astimezone(MOSCOW_TZ)
            rows.append({
                "datetime":          ts,
                "fear_greed_value":  int(item["value"]),
                "fear_greed_label":  item["value_classification"],
            })
        df = pd.DataFrame(rows).set_index("datetime")
        print(f"  Fear & Greed: {len(df)} записей")
        return df
    except Exception as e:
        print(f"  Fear & Greed: ошибка — {e}")
        return pd.DataFrame()

def merge_and_save(btc: pd.DataFrame, eth: pd.DataFrame, fg: pd.DataFrame):
    if btc.empty or eth.empty:
        print("Нет данных для записи!")
        return

    df = btc.join(eth, how="outer")

    if not fg.empty:
        fg_hourly = fg.reindex(df.index, method="ffill")
        df = df.join(fg_hourly)
    else:
        df["fear_greed_value"] = None
        df["fear_greed_label"] = None

    df["media_sentiment"]      = None
    df["media_articles_count"] = None
    df["btc_oi_usd"]           = None
    df["eth_oi_usd"]           = None

    conn = sqlite3.connect(DB_PATH)
    inserted = 0
    skipped  = 0

    for ts, row in df.iterrows():
        timestamp = ts.strftime("%Y-%m-%d %H:%M")

        btc_price = row.get("btc_price")
        eth_price = row.get("eth_price")
        if pd.isna(btc_price) or pd.isna(eth_price):
            skipped += 1
            continue

        try:
            conn.execute("""
                INSERT OR IGNORE INTO hourly_snapshots (
                    timestamp, btc_price, eth_price,
                    btc_atr_pct, eth_atr_pct,
                    fear_greed_value, fear_greed_label,
                    media_sentiment, media_articles_count,
                    btc_oi_usd, eth_oi_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                float(btc_price),
                float(eth_price),
                float(row["btc_atr_pct"]) if not pd.isna(row.get("btc_atr_pct", float("nan"))) else None,
                float(row["eth_atr_pct"]) if not pd.isna(row.get("eth_atr_pct", float("nan"))) else None,
                int(row["fear_greed_value"]) if row.get("fear_greed_value") is not None and not pd.isna(row["fear_greed_value"]) else None,
                str(row["fear_greed_label"]) if row.get("fear_greed_label") is not None else None,
                None, None, None, None,
            ))
            inserted += 1
        except Exception as e:
            print(f"  Ошибка записи {timestamp}: {e}")

    conn.commit()
    conn.close()
    print(f"\nГотово! Записано: {inserted}, пропущено: {skipped}")
    print(f"БД: {DB_PATH.absolute()}")

def main():
    print(f"=== Заполнение истории за {DAYS_BACK} дней ===\n")
    init_db()

    btc = fetch_prices_and_atr("BTC", "BTC-USD")
    eth = fetch_prices_and_atr("ETH", "ETH-USD")
    fg  = fetch_fear_greed(DAYS_BACK)

    merge_and_save(btc, eth, fg)

    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM hourly_snapshots").fetchone()[0]
    first = conn.execute("SELECT MIN(timestamp) FROM hourly_snapshots").fetchone()[0]
    last  = conn.execute("SELECT MAX(timestamp) FROM hourly_snapshots").fetchone()[0]
    conn.close()
    print(f"\nВ БД всего записей: {count}")
    print(f"Период: {first} → {last}")

if __name__ == "__main__":
    main()
