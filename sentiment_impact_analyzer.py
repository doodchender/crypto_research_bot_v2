import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pytz

MOSCOW_TZ = pytz.timezone("Europe/Moscow")
CATEGORIES_FILE = Path("data/sentiment_categories.json")

HORIZON_DAYS = 1

MIN_SAMPLES = 20

BINS = [-1.0, -0.3, -0.1, 0.1, 0.3, 1.0]
BIN_LABELS = ["сильный негатив", "умеренный негатив", "нейтральный",
              "умеренный позитив", "сильный позитив"]
BIN_CATEGORIES = [-2, -1, 0, 1, 2]

def load_historical_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    needed = ["date", "media_sentiment", "log_return", "high_vol", "bull", "price_usd"]
    df = df[[c for c in needed if c in df.columns]].dropna(subset=["media_sentiment", "log_return"])
    df = df.rename(columns={"log_return": "price_change", "media_sentiment": "sentiment"})
    df["source"] = "historical"
    return df

def load_live_data(db_path: str, days_back: int = 90) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        cutoff = (datetime.now(MOSCOW_TZ) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT timestamp, media_sentiment as sentiment,
                   btc_price, eth_price
            FROM hourly_snapshots
            WHERE timestamp >= ? AND media_sentiment IS NOT NULL
            ORDER BY timestamp
        """, (cutoff,)).fetchall()
        conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "sentiment", "btc_price", "eth_price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp")
        df["price_change"] = df["btc_price"].pct_change(24).fillna(0)
        df["high_vol"] = 0
        df["bull"] = 1
        df["source"] = "live"
        return df[["timestamp", "sentiment", "price_change", "high_vol", "bull", "source"]]
    except Exception as e:
        print(f"Live data error: {e}")
        return pd.DataFrame()

def compute_bin_stats(values: np.ndarray) -> dict:
    if len(values) < 3:
        return {}
    return {
        "count":        int(len(values)),
        "mean":         float(np.mean(values)),
        "median":       float(np.median(values)),
        "std":          float(np.std(values)),
        "p10":          float(np.percentile(values, 10)),
        "p75":          float(np.percentile(values, 75)),
        "p90":          float(np.percentile(values, 90)),
        "prob_up":      float(np.mean(values > 0)),
        "prob_up_1pct": float(np.mean(values > 0.01)),
        "prob_dn_1pct": float(np.mean(values < -0.01)),
        "prob_up_5pct": float(np.mean(values > 0.05)),
        "prob_dn_5pct": float(np.mean(values < -0.05)),
    }

def determine_category(stats: dict) -> int:
    if not stats:
        return 0
    median = stats["median"]
    prob_up = stats["prob_up"]

    if median > 0.03 and prob_up > 0.65:
        return 2
    elif median > 0.01 and prob_up > 0.55:
        return 1
    elif median < -0.03 and prob_up < 0.35:
        return -2
    elif median < -0.01 and prob_up < 0.45:
        return -1
    else:
        return 0

def analyze(df: pd.DataFrame, symbol: str) -> dict:
    if df.empty or "sentiment" not in df.columns:
        return {}

    result = {"bins": [], "updated": datetime.now(MOSCOW_TZ).isoformat(), "n_total": len(df)}

    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        mask = (df["sentiment"] >= lo) & (df["sentiment"] < hi)
        subset = df[mask]["price_change"].values

        stats = compute_bin_stats(subset)
        cat   = determine_category(stats) if len(subset) >= MIN_SAMPLES else None

        result["bins"].append({
            "min":        lo,
            "max":        hi,
            "label":      BIN_LABELS[i],
            "category":   cat,
            "reliable":   len(subset) >= MIN_SAMPLES,
            "stats":      stats,
        })

    return result

def run_update(cfg: dict = None, logger: logging.Logger = None):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("analyzer")

    logger.info("Запуск sentiment_impact_analyzer...")

    ofefki_path = Path(cfg.get("ofefki_path", "../ofefki") if cfg else "../ofefki")
    db_path     = Path(cfg.get("paths", {}).get("data", "data") if cfg else "data") / "hourly.db"

    btc_csv = ofefki_path / "data/processed/final_data_btc.csv"
    eth_csv = ofefki_path / "data/processed/final_data_eth.csv"

    btc_hist = load_historical_csv(btc_csv)
    eth_hist = load_historical_csv(eth_csv)
    logger.info(f"Исторические данные: BTC={len(btc_hist)}, ETH={len(eth_hist)} строк")

    live = load_live_data(str(db_path))
    logger.info(f"Живые данные: {len(live)} строк")

    btc_df = pd.concat([btc_hist, live], ignore_index=True) if not live.empty else btc_hist
    eth_df = pd.concat([eth_hist, live], ignore_index=True) if not live.empty else eth_hist

    categories = {
        "btc": analyze(btc_df, "BTC"),
        "eth": analyze(eth_df, "ETH"),
    }

    CATEGORIES_FILE.parent.mkdir(exist_ok=True)
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    logger.info(f"Категории сохранены: {CATEGORIES_FILE}")

    for symbol, data in categories.items():
        bins = data.get("bins", [])
        logger.info(f"{symbol.upper()}: {data.get('n_total', 0)} наблюдений, {len(bins)} бинов")
        for b in bins:
            cat_str = str(b['category']) if b['category'] is not None else "N/A"
            logger.info(
                f"  [{b['min']:.1f}, {b['max']:.1f}] {b['label']}: "
                f"n={b['stats'].get('count', 0)}, cat={cat_str}, "
                f"median={b['stats'].get('median', 0):.3f}"
            )

    return categories

def load_categories() -> dict:
    if not CATEGORIES_FILE.exists():
        return {}
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_forecast(sentiment_value: float, symbol: str = "btc",
                 high_vol: bool = False) -> dict:
    cats = load_categories()
    data = cats.get(symbol.lower(), {})
    bins = data.get("bins", [])

    if not bins:
        return {"error": "Категории не рассчитаны"}

    matched = None
    for b in bins:
        if b["min"] <= sentiment_value < b["max"]:
            matched = b
            break

    if matched is None and sentiment_value >= BINS[-2]:
        matched = bins[-1]

    if matched is None:
        return {"error": "Сентимент вне диапазона"}

    if not matched.get("reliable"):
        return {
            "error": f"Недостаточно данных для бина «{matched['label']}» "
                     f"(нужно ≥{MIN_SAMPLES}, есть {matched['stats'].get('count', 0)})"
        }

    stats = matched["stats"]
    result = {
        "sentiment":   sentiment_value,
        "label":       matched["label"],
        "category":    matched["category"],
        "median_chg":  stats.get("median", 0),
        "p10":         stats.get("p10", 0),
        "p25":         stats.get("p25", 0),
        "p75":         stats.get("p75", 0),
        "p90":         stats.get("p90", 0),
        "prob_up":     stats.get("prob_up", 0.5),
        "prob_up_1":   stats.get("prob_up_1pct", 0),
        "prob_dn_1":   stats.get("prob_dn_1pct", 0),
        "count":       stats.get("count", 0),
        "high_vol":    high_vol,
    }
    return result

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    run_update(cfg)
