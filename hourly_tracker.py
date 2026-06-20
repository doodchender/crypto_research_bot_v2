import asyncio
import logging
from datetime import datetime
import pytz

from database import save_snapshot, cleanup_old_records
from collectors.prices         import get_live_prices
from collectors.fear_greed     import get_fear_greed
from collectors.atr_signals    import get_atr_signals
from collectors.social_sentiment import get_social_sentiment as get_rss_sentiment
from collectors.coinglass_data import get_futures_data

MOSCOW_TZ = pytz.timezone("Europe/Moscow")

VALIDATION = {
    "btc_price":    (1_000,   500_000),
    "eth_price":    (10,      50_000),
    "btc_atr_pct":  (0.1,    30.0),
    "eth_atr_pct":  (0.1,    30.0),
    "fear_greed_value": (0,  100),
    "media_sentiment":  (-1.0, 1.0),
}

def _validate(key: str, value) -> bool:
    if value is None:
        return True
    bounds = VALIDATION.get(key)
    if not bounds:
        return True
    return bounds[0] <= value <= bounds[1]

async def collect_snapshot(cfg: dict, logger: logging.Logger) -> dict:
    now = datetime.now(MOSCOW_TZ)
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    logger.info(f"Hourly snapshot: {timestamp}")

    loop = asyncio.get_running_loop()

    results = await asyncio.gather(
        loop.run_in_executor(None, lambda: get_live_prices(logger)),
        loop.run_in_executor(None, lambda: get_atr_signals(logger)),
        loop.run_in_executor(None, lambda: get_fear_greed(cfg, logger)),
        loop.run_in_executor(None, lambda: get_rss_sentiment(logger)),
        loop.run_in_executor(None, lambda: get_futures_data(cfg, logger)),
        return_exceptions=True
    )

    prices, atr, fg, sentiment, futures = results

    btc_price = eth_price = None
    if isinstance(prices, dict):
        btc = prices.get("BTC")
        eth = prices.get("ETH")
        btc_price = btc.get("price") if btc else None
        eth_price = eth.get("price") if eth else None

    btc_atr_pct = eth_atr_pct = None
    if isinstance(atr, dict):
        b = atr.get("BTC")
        e = atr.get("ETH")
        btc_atr_pct = b.get("atr_pct") if b else None
        eth_atr_pct = e.get("atr_pct") if e else None

    fear_greed_value = fear_greed_label = None
    if isinstance(fg, dict) and fg:
        fear_greed_value = fg.get("value")
        fear_greed_label = fg.get("label")

    media_sentiment = media_articles_count = None
    if isinstance(sentiment, dict) and sentiment:
        media_sentiment      = sentiment.get("avg_sentiment")
        media_articles_count = sentiment.get("count")

    btc_oi_usd = eth_oi_usd = None
    if isinstance(futures, dict) and cfg.get("track_oi", True):
        b_oi = futures.get("BTC", {}).get("open_interest")
        e_oi = futures.get("ETH", {}).get("open_interest")
        btc_oi_usd = b_oi.get("oi_usd") if b_oi else None
        eth_oi_usd = e_oi.get("oi_usd") if e_oi else None

    snapshot = {
        "timestamp":            timestamp,
        "btc_price":            btc_price   if _validate("btc_price",   btc_price)   else None,
        "eth_price":            eth_price   if _validate("eth_price",   eth_price)   else None,
        "btc_atr_pct":          btc_atr_pct if _validate("btc_atr_pct", btc_atr_pct) else None,
        "eth_atr_pct":          eth_atr_pct if _validate("eth_atr_pct", eth_atr_pct) else None,
        "fear_greed_value":     fear_greed_value if _validate("fear_greed_value", fear_greed_value) else None,
        "fear_greed_label":     fear_greed_label,
        "media_sentiment":      media_sentiment if _validate("media_sentiment", media_sentiment) else None,
        "media_articles_count": media_articles_count,
        "btc_oi_usd":           btc_oi_usd,
        "eth_oi_usd":           eth_oi_usd,
    }

    logger.info(
        f"Snapshot: BTC=${snapshot['btc_price']}, ETH=${snapshot['eth_price']}, "
        f"FG={snapshot['fear_greed_value']}, Sent={snapshot['media_sentiment']}"
    )

    return snapshot

async def run_hourly_snapshot(context):
    cfg    = context.bot_data["cfg"]
    logger = context.bot_data["logger"]

    try:
        snapshot = await collect_snapshot(cfg, logger)
        save_snapshot(cfg, logger, snapshot)
        now = datetime.now(MOSCOW_TZ)
        if now.hour == 3:
            cleanup_old_records(cfg, logger)
    except Exception as e:
        logger.error(f"Hourly snapshot failed: {e}", exc_info=True)
