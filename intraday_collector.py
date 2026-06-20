import asyncio
import logging
import sqlite3
import math
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pytz

MOSCOW_TZ = pytz.timezone("Europe/Moscow")
INTERVAL_SEC = 300
RETENTION_DAYS = 30
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

REST_PRICE      = "https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
REST_OI         = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
REST_DEPTH      = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
REST_KLINE      = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=5"
REST_FUNDING    = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
REST_TAKER_VOL  = "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=1"
REST_LS_RATIO   = "https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
REST_STABLECOIN  = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
REST_DERIBIT_OPT = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=option"
REST_BYBIT_PRICE = "https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
REST_GAS_FEES    = "https://api.etherscan.io/v2/api?chainid=1&module=gastracker&action=gasoracle&apikey=QJJHXJMH9TWF1T2KPFAETMAQ3VBGSBIC1F"
REST_COINGECKO   = "https://api.coingecko.com/api/v3/global"
REST_WHALE_BTC   = "https://blockchain.info/unconfirmed-transactions?format=json"
REST_MEMPOOL     = "https://blockchain.info/q/unconfirmedcount"
REST_BTC_STATS   = "https://blockchain.info/stats?format=json"
REST_SPOT_24H    = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
REST_AGG_TRADES  = "https://api.binance.com/api/v3/aggTrades?symbol={symbol}&limit=500"
WS_LIQUIDATIONS  = "wss://fstream.binance.com/ws/!forceOrder@arr"

_stablecoin_cache = {"usdt_circ": 0, "usdc_circ": 0, "usdt_delta_24h": 0, "usdc_delta_24h": 0}
_stock_cache = {"sp500": 0.0, "nasdaq": 0.0, "sp500_ret": 0.0, "nasdaq_ret": 0.0, "ts": 0}

_running = False
_vol_estimators = {}
_alert_callback = None
_last_alert_ts = {}

_liq_buffer: list = []
_liq_buffer_lock = None

def set_alert_callback(callback):
    global _alert_callback
    _alert_callback = callback

def _send_alert(level: str, key: str, message: str, cooldown_sec: int = 600):
    if _alert_callback is None:
        return
    import time as _time
    now = _time.time()
    last = _last_alert_ts.get(key, 0)
    if now - last < cooldown_sec:
        return
    try:
        _alert_callback(level, message)
        _last_alert_ts[key] = now
    except Exception:
        pass

def init_intraday_db(db_path: str, logger: logging.Logger):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intraday_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL UNIQUE,   -- Moscow time YYYY-MM-DD HH:MM
            btc_price       REAL,
            eth_price       REAL,
            btc_oi          REAL,
            eth_oi          REAL,
            btc_ob_imb      REAL,   -- order book imbalance 0..1
            eth_ob_imb      REAL,
            btc_spread_pct  REAL,
            eth_spread_pct  REAL,
            btc_volume_5m   REAL,   -- объём торгов за 5 минут (USD)
            eth_volume_5m   REAL,
            btc_whale_buy   REAL,   -- whale покупки за 5 минут
            btc_whale_sell  REAL,
            eth_whale_buy   REAL,
            eth_whale_sell  REAL,
            btc_ret_5m      REAL,   -- log return за 5 минут (заполняется позже)
            btc_ret_15m     REAL,   -- log return за 15 минут
            btc_ret_30m     REAL,   -- log return за 30 минут
            eth_ret_5m      REAL,
            eth_ret_15m     REAL,
            eth_ret_30m     REAL,
            btc_funding     REAL,   -- текущий funding rate BTC
            eth_funding     REAL,   -- текущий funding rate ETH
            usdt_circ       REAL,   -- USDT в обращении (USD)
            usdc_circ       REAL,   -- USDC в обращении (USD)
            usdt_delta_24h  REAL,   -- изменение USDT за 24h (мин/сожж)
            usdc_delta_24h  REAL,   -- изменение USDC за 24h
            btc_taker_buy_vol  REAL,  -- taker buy volume 5m
            btc_taker_sell_vol REAL,
            eth_taker_buy_vol  REAL,
            eth_taker_sell_vol REAL,
            btc_ls_ratio    REAL,   -- long/short ratio
            eth_ls_ratio    REAL,
            btc_depth_imb_5 REAL,   -- order book depth imbalance (5 lvl)
            btc_depth_imb_20 REAL,  -- order book depth imbalance (20 lvl)
            eth_depth_imb_5 REAL,
            eth_depth_imb_20 REAL,
            btc_pc_vol_ratio REAL,  -- put/call volume ratio
            btc_pc_oi_ratio  REAL,  -- put/call open interest ratio
            eth_pc_vol_ratio REAL,
            eth_pc_oi_ratio  REAL,
            sol_price        REAL,  -- цена SOL (для Granger SOL→ETH)
            btc_bybit_price  REAL,  -- цена BTC на Bybit (cross-exchange)
            eth_bybit_price  REAL,  -- цена ETH на Bybit
            gas_fee          REAL,  -- gas fees в gwei (Ethereum)
            btc_gex          REAL,  -- Gamma Exposure BTC (млрд)
            eth_gex          REAL,  -- Gamma Exposure ETH
            btc_options_skew REAL,  -- IV skew: OTM put IV - OTM call IV
            eth_options_skew REAL,
            btc_max_pain     REAL,  -- Max Pain отклонение от спота (%)
            eth_max_pain     REAL,
            bnb_price        REAL,  -- цена BNB (активность Binance)
            xrp_price        REAL,  -- цена XRP (ротация капитала)
            link_price       REAL,  -- цена LINK (DeFi/ETH экосистема)
            btc_dominance    REAL,  -- BTC.D доминанс в % (CoinGecko)
            social_sentiment REAL,  -- сентимент соцсетей (-1..+1)
            social_volume    INTEGER, -- кол-во постов
            trends_interest  REAL,  -- Google Trends bitcoin (0-100)
            whale_tx_count   INTEGER, -- кол-во крупных BTC транзакций (>10 BTC)
            whale_tx_volume  REAL,  -- объём крупных транзакций в BTC
            sp500_price      REAL,  -- S&P500 цена
            nasdaq_price     REAL,  -- NASDAQ цена
            sp500_ret_5m     REAL,  -- S&P500 return за 5 мин
            nasdaq_ret_5m    REAL,  -- NASDAQ return за 5 мин
            mempool_size     INTEGER, -- кол-во неподтверждённых транзакций
            active_addresses INTEGER, -- активные адреса за сутки
            btc_tx_volume    REAL,  -- объём BTC транзакций on-chain
            btc_wmid_dev     REAL,  -- weighted mid-price deviation (bps)
            eth_wmid_dev     REAL,
            btc_price_impact REAL,  -- USD to move price 0.1%
            eth_price_impact REAL,
            btc_queue_imb    REAL,  -- order count imbalance
            eth_queue_imb    REAL,
            realized_cap     REAL,  -- BTC realized cap
            mempool_fee      REAL   -- средняя комиссия mempool (sat/byte)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_ts ON intraday_snapshots(timestamp)")

    existing = {row[1] for row in conn.execute("PRAGMA table_info(intraday_snapshots)")}
    new_cols = [
        ("btc_funding", "REAL"),
        ("eth_funding", "REAL"),
        ("usdt_circ", "REAL"),
        ("usdc_circ", "REAL"),
        ("usdt_delta_24h", "REAL"),
        ("usdc_delta_24h", "REAL"),
        ("btc_taker_buy_vol", "REAL"),
        ("btc_taker_sell_vol", "REAL"),
        ("eth_taker_buy_vol", "REAL"),
        ("eth_taker_sell_vol", "REAL"),
        ("btc_ls_ratio", "REAL"),
        ("eth_ls_ratio", "REAL"),
        ("btc_depth_imb_5", "REAL"),
        ("btc_depth_imb_20", "REAL"),
        ("eth_depth_imb_5", "REAL"),
        ("eth_depth_imb_20", "REAL"),
        ("btc_pc_vol_ratio", "REAL"),
        ("btc_pc_oi_ratio", "REAL"),
        ("eth_pc_vol_ratio", "REAL"),
        ("eth_pc_oi_ratio", "REAL"),
        ("sol_price", "REAL"),
        ("btc_bybit_price", "REAL"),
        ("eth_bybit_price", "REAL"),
        ("gas_fee", "REAL"),
        ("btc_gex", "REAL"),
        ("eth_gex", "REAL"),
        ("btc_options_skew", "REAL"),
        ("eth_options_skew", "REAL"),
        ("btc_max_pain", "REAL"),
        ("eth_max_pain", "REAL"),
        ("bnb_price", "REAL"),
        ("xrp_price", "REAL"),
        ("link_price", "REAL"),
        ("btc_dominance", "REAL"),
        ("social_sentiment", "REAL"),
        ("social_volume", "INTEGER"),
        ("trends_interest", "REAL"),
        ("whale_tx_count", "INTEGER"),
        ("whale_tx_volume", "REAL"),
        ("sp500_price", "REAL"),
        ("nasdaq_price", "REAL"),
        ("sp500_ret_5m", "REAL"),
        ("nasdaq_ret_5m", "REAL"),
        ("mempool_size", "INTEGER"),
        ("active_addresses", "INTEGER"),
        ("btc_tx_volume", "REAL"),
        ("btc_wmid_dev", "REAL"),
        ("eth_wmid_dev", "REAL"),
        ("btc_price_impact", "REAL"),
        ("eth_price_impact", "REAL"),
        ("btc_queue_imb", "REAL"),
        ("eth_queue_imb", "REAL"),
        ("realized_cap", "REAL"),
        ("mempool_fee", "REAL"),
        ("btc_egarch_vol", "REAL"),
        ("eth_egarch_vol", "REAL"),
        ("btc_vol_anomaly", "INTEGER"),
        ("eth_vol_anomaly", "INTEGER"),
        ("btc_spot_vol_24h", "REAL"),
        ("eth_spot_vol_24h", "REAL"),
        ("btc_leverage_ratio", "REAL"),
        ("eth_leverage_ratio", "REAL"),
        ("btc_oi_velocity_1h", "REAL"),
        ("eth_oi_velocity_1h", "REAL"),
        ("btc_cascade_risk", "REAL"),
        ("eth_cascade_risk", "REAL"),
        ("btc_long_liq_usd",  "REAL"),
        ("btc_short_liq_usd", "REAL"),
        ("btc_liq_imb",       "REAL"),
        ("btc_liq_total_usd", "REAL"),
        ("eth_long_liq_usd",  "REAL"),
        ("eth_short_liq_usd", "REAL"),
        ("eth_liq_imb",       "REAL"),
        ("eth_liq_total_usd", "REAL"),
        ("btc_large_buy_usd",    "REAL"),
        ("btc_large_sell_usd",   "REAL"),
        ("btc_large_trade_imb",  "REAL"),
        ("btc_large_trade_count","INTEGER"),
        ("eth_large_buy_usd",    "REAL"),
        ("eth_large_sell_usd",   "REAL"),
        ("eth_large_trade_imb",  "REAL"),
        ("eth_large_trade_count","INTEGER"),
        ("btc_funding_delta", "REAL"),
        ("eth_funding_delta", "REAL"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE intraday_snapshots ADD COLUMN {col_name} {col_type}")
            logger.info(f"Добавлена колонка: {col_name}")

    conn.commit()
    conn.close()
    logger.info(f"Intraday DB инициализирована: {db_path}")

def cleanup_intraday(db_path: str, logger: logging.Logger):
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(db_path)
        n = conn.execute("DELETE FROM intraday_snapshots WHERE timestamp < ?", (cutoff,)).rowcount
        conn.commit()
        conn.close()
        if n:
            logger.info(f"Intraday cleanup: удалено {n} записей")
    except Exception as e:
        logger.error(f"Intraday cleanup error: {e}")

def backfill_returns(db_path: str, logger: logging.Logger):
    try:
        conn = sqlite3.connect(db_path)

        rows = conn.execute("""
            SELECT id, timestamp, btc_price, eth_price
            FROM intraday_snapshots
            WHERE (btc_ret_5m IS NULL OR btc_ret_15m IS NULL OR btc_ret_30m IS NULL)
              AND btc_price IS NOT NULL
            ORDER BY timestamp
        """).fetchall()

        updated = 0
        for row_id, ts, btc_now, eth_now in rows:
            if not btc_now or not eth_now:
                continue

            updates = {}
            for horizon, col_btc, col_eth in [
                (1, "btc_ret_5m",  "eth_ret_5m"),
                (3, "btc_ret_15m", "eth_ret_15m"),
                (6, "btc_ret_30m", "eth_ret_30m"),
            ]:
                future = conn.execute("""
                    SELECT btc_price, eth_price FROM intraday_snapshots
                    WHERE timestamp > ? AND btc_price IS NOT NULL
                    ORDER BY timestamp LIMIT 1 OFFSET ?
                """, (ts, horizon - 1)).fetchone()

                if future:
                    btc_f, eth_f = future
                    if btc_f and btc_now:
                        updates[col_btc] = math.log(btc_f / btc_now)
                    if eth_f and eth_now:
                        updates[col_eth] = math.log(eth_f / eth_now)

            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                conn.execute(
                    f"UPDATE intraday_snapshots SET {set_clause} WHERE id = ?",
                    list(updates.values()) + [row_id]
                )
                updated += 1

        conn.commit()
        conn.close()
        if updated:
            logger.debug(f"Backfill returns: обновлено {updated} записей")
    except Exception as e:
        logger.error(f"Backfill error: {e}")

async def _retry_get(session: aiohttp.ClientSession, url: str,
                     retries: int = 3, timeout: float = 5.0):
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status == 200:
                    return await r.json(content_type=None)
                elif r.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
        if attempt < retries - 1:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None

async def fetch_price(session: aiohttp.ClientSession, symbol: str):
    data = await _retry_get(session, REST_PRICE.format(symbol=symbol))
    if data is None:
        return None
    try:
        price = float(data.get("price", 0))
        return price if price > 0 else None
    except (ValueError, TypeError):
        return None

async def fetch_oi(session: aiohttp.ClientSession, symbol: str):
    data = await _retry_get(session, REST_OI.format(symbol=symbol))
    if data is None:
        return None
    try:
        oi = float(data.get("openInterest", 0))
        return oi if oi > 0 else None
    except (ValueError, TypeError):
        return None

async def fetch_orderbook(session: aiohttp.ClientSession, symbol: str) -> dict:
    try:
        url = REST_DEPTH.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return {}
        bid_price = float(bids[0][0])
        ask_price = float(asks[0][0])
        bid_vol = sum(float(b[0]) * float(b[1]) for b in bids)
        ask_vol = sum(float(a[0]) * float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        imbalance  = bid_vol / total if total > 0 else 0.5
        spread_pct = (ask_price - bid_price) / bid_price * 100 if bid_price > 0 else 0
        return {"imbalance": imbalance, "spread_pct": spread_pct}
    except Exception:
        return {}

async def fetch_volume_5m(session: aiohttp.ClientSession, symbol: str) -> float:
    try:
        url = REST_KLINE.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
        total_vol = sum(float(k[7]) for k in data)
        return total_vol
    except Exception:
        return 0.0

def get_whale_flow_5m(whale_db: str) -> dict:
    result = {s: {"buy": 0.0, "sell": 0.0} for s in SYMBOLS}
    if not Path(whale_db).exists():
        return result
    try:
        cutoff = (datetime.now(MOSCOW_TZ) - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(whale_db)
        rows = conn.execute("""
            SELECT symbol, side, SUM(volume_usd)
            FROM whale_trades
            WHERE timestamp >= ?
            GROUP BY symbol, side
        """, (cutoff,)).fetchall()
        conn.close()
        for symbol, side, vol in rows:
            if symbol in result and vol:
                result[symbol][side] = vol
    except Exception:
        pass
    return result

async def fetch_funding_rate(session: aiohttp.ClientSession, symbol: str) -> float:
    try:
        url = REST_FUNDING.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
            return float(data.get("lastFundingRate", 0))
    except Exception:
        return 0.0

async def fetch_taker_volume(session: aiohttp.ClientSession, symbol: str) -> dict:
    try:
        url = REST_TAKER_VOL.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
            if data:
                return {
                    "buy_vol": float(data[0].get("buyVol", 0)),
                    "sell_vol": float(data[0].get("sellVol", 0)),
                    "ratio": float(data[0].get("buySellRatio", 1)),
                }
    except Exception:
        pass
    return {"buy_vol": 0, "sell_vol": 0, "ratio": 1.0}

async def fetch_ls_ratio(session: aiohttp.ClientSession, symbol: str) -> float:
    try:
        url = REST_LS_RATIO.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
            if data:
                return float(data[0].get("longShortRatio", 1))
    except Exception:
        pass
    return 1.0

async def fetch_orderbook_depth(session: aiohttp.ClientSession, symbol: str) -> dict:
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return {}

        bid_vol_5 = sum(float(b[0]) * float(b[1]) for b in bids[:5])
        ask_vol_5 = sum(float(a[0]) * float(a[1]) for a in asks[:5])
        total_5 = bid_vol_5 + ask_vol_5
        depth_imb_5 = (bid_vol_5 - ask_vol_5) / total_5 if total_5 > 0 else 0

        bid_vol_20 = sum(float(b[0]) * float(b[1]) for b in bids[:20])
        ask_vol_20 = sum(float(a[0]) * float(a[1]) for a in asks[:20])
        total_20 = bid_vol_20 + ask_vol_20
        depth_imb_20 = (bid_vol_20 - ask_vol_20) / total_20 if total_20 > 0 else 0

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        bid_vol_1 = float(bids[0][0]) * float(bids[0][1])
        ask_vol_1 = float(asks[0][0]) * float(asks[0][1])
        total_1 = bid_vol_1 + ask_vol_1
        wmid = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / total_1 if total_1 > 0 else (best_bid + best_ask) / 2
        mid = (best_bid + best_ask) / 2
        wmid_dev = (wmid - mid) / mid * 10000 if mid > 0 else 0

        target_move = mid * 0.001
        buy_impact = 0.0
        cumul = 0.0
        for a in asks:
            price, qty = float(a[0]), float(a[1])
            if price > mid + target_move:
                break
            cumul += price * qty
        buy_impact = cumul if cumul > 0 else 0.0

        bid_count = len(bids)
        ask_count = len(asks)
        queue_imb = (bid_count - ask_count) / (bid_count + ask_count) if (bid_count + ask_count) > 0 else 0

        return {
            "depth_imb_5": depth_imb_5,
            "depth_imb_20": depth_imb_20,
            "wmid_dev": wmid_dev,
            "price_impact": buy_impact,
            "queue_imb": queue_imb,
        }
    except Exception:
        return {}

def _bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        nd1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        return nd1 / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0

async def fetch_deribit_options(session: aiohttp.ClientSession,
                                 currency: str,
                                 logger: logging.Logger) -> dict:
    default = {
        "pc_vol_ratio": 1.0, "pc_oi_ratio": 1.0,
        "skew": 0.0, "gex": 0.0, "max_pain": 0.0,
    }
    try:
        url = REST_DERIBIT_OPT.format(currency=currency)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
        options = data.get("result", [])
        if not options:
            return default

        spot = next((float(o["underlying_price"]) for o in options
                     if o.get("underlying_price")), 0.0)
        if spot <= 0:
            return default

        MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

        parsed = []
        for o in options:
            name = o.get("instrument_name", "")
            parts = name.split("-")
            if len(parts) != 4:
                continue
            try:
                strike   = float(parts[2])
                opt_type = parts[3]
                exp_str  = parts[1]
                mark_iv  = float(o.get("mark_iv") or 0)
                oi       = float(o.get("open_interest") or 0)
                vol_usd  = float(o.get("volume_usd") or 0)

                day  = int(exp_str[:2])
                mon  = MONTHS.get(exp_str[2:5], 1)
                year = 2000 + int(exp_str[5:])
                expiry_dt = datetime(year, mon, day, 8, 0)
                now_utc   = datetime.utcnow()
                T = max((expiry_dt - now_utc).total_seconds() / (365.25 * 86400), 1/365)

                parsed.append({
                    "type": opt_type, "strike": strike,
                    "iv": mark_iv / 100, "oi": oi, "vol_usd": vol_usd,
                    "T": T, "moneyness": strike / spot,
                })
            except (ValueError, KeyError):
                continue

        puts  = [p for p in parsed if p["type"] == "P"]
        calls = [p for p in parsed if p["type"] == "C"]

        put_vol  = sum(p["vol_usd"] for p in puts)
        call_vol = sum(c["vol_usd"] for c in calls)
        put_oi   = sum(p["oi"] for p in puts)
        call_oi  = sum(c["oi"] for c in calls)
        pc_vol = put_vol  / call_vol  if call_vol  > 0 else 1.0
        pc_oi  = put_oi   / call_oi   if call_oi   > 0 else 1.0

        otm_puts  = [p for p in puts  if 0.85 <= p["moneyness"] <= 0.97 and p["iv"] > 0]
        otm_calls = [c for c in calls if 1.03 <= c["moneyness"] <= 1.15 and c["iv"] > 0]
        if otm_puts and otm_calls:
            skew = (sum(p["iv"] for p in otm_puts)  / len(otm_puts) -
                    sum(c["iv"] for c in otm_calls) / len(otm_calls))
        else:
            skew = 0.0

        gex = 0.0
        near = [o for o in parsed if o["T"] < 30/365 and o["iv"] > 0 and o["oi"] > 0]
        for o in near:
            g = _bs_gamma(spot, o["strike"], o["T"], o["iv"])
            sign = 1.0 if o["type"] == "C" else -1.0
            gex += sign * g * o["oi"] * spot ** 2 / 1e9

        max_pain_pct = 0.0
        near_exp = [o for o in parsed if o["T"] < 45/365 and o["oi"] > 0]
        if near_exp:
            strikes = sorted({o["strike"] for o in near_exp})
            by_strike = {}
            for o in near_exp:
                k = o["strike"]
                if k not in by_strike:
                    by_strike[k] = {"C": 0.0, "P": 0.0}
                by_strike[k][o["type"]] += o["oi"]

            best_k, best_pain = spot, float("inf")
            for test_k in strikes:
                pain = sum(
                    v["C"] * max(test_k - k, 0) + v["P"] * max(k - test_k, 0)
                    for k, v in by_strike.items()
                )
                if pain < best_pain:
                    best_pain, best_k = pain, test_k
            max_pain_pct = (best_k - spot) / spot * 100

        return {
            "pc_vol_ratio": pc_vol, "pc_oi_ratio": pc_oi,
            "skew": skew, "gex": gex, "max_pain": max_pain_pct,
        }

    except Exception as e:
        logger.debug(f"Deribit options error ({currency}): {e}")
        return default

async def fetch_stablecoin_flows(session: aiohttp.ClientSession,
                                  logger: logging.Logger) -> dict:
    global _stablecoin_cache
    try:
        async with session.get(
            REST_STABLECOIN,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            data = await r.json()

        for coin in data.get("peggedAssets", []):
            symbol = coin.get("symbol", "")
            if symbol in ("USDT", "USDC"):
                circ = coin.get("circulating", {}).get("peggedUSD", 0) or 0
                prev_day = coin.get("circulatingPrevDay", {}).get("peggedUSD", 0) or 0
                delta = circ - prev_day if prev_day else 0

                if symbol == "USDT":
                    _stablecoin_cache["usdt_circ"] = circ
                    _stablecoin_cache["usdt_delta_24h"] = delta
                elif symbol == "USDC":
                    _stablecoin_cache["usdc_circ"] = circ
                    _stablecoin_cache["usdc_delta_24h"] = delta

    except Exception as e:
        logger.debug(f"Stablecoin fetch error: {e}")

    return _stablecoin_cache

async def fetch_bybit_price(session: aiohttp.ClientSession, symbol: str) -> float:
    try:
        url = REST_BYBIT_PRICE.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
        items = data.get("result", {}).get("list", [])
        if items:
            return float(items[0].get("lastPrice", 0))
    except Exception:
        pass
    return 0.0

async def fetch_sol_price(session: aiohttp.ClientSession) -> float:
    try:
        url = REST_PRICE.format(symbol="SOLUSDT")
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
            return float(data.get("price", 0))
    except Exception:
        return 0.0

async def fetch_alt_prices(session: aiohttp.ClientSession) -> dict:
    result = {"bnb": 0.0, "xrp": 0.0, "link": 0.0}
    for sym, key in [("BNBUSDT", "bnb"), ("XRPUSDT", "xrp"), ("LINKUSDT", "link")]:
        try:
            url = REST_PRICE.format(symbol=sym)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
                result[key] = float(data.get("price", 0))
        except Exception:
            pass
    return result

async def fetch_btc_dominance(session: aiohttp.ClientSession,
                               logger: logging.Logger) -> float:
    try:
        async with session.get(
            REST_COINGECKO,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            data = await r.json()
        return float(data.get("data", {}).get("btc_dominance", 0))
    except Exception as e:
        logger.debug(f"BTC dominance fetch error: {e}")
        return 0.0

async def fetch_whale_transactions(session: aiohttp.ClientSession,
                                     logger: logging.Logger) -> dict:
    try:
        async with session.get(
            REST_WHALE_BTC,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            data = await r.json(content_type=None)
        txs = data.get("txs", [])
        threshold = 10 * 100_000_000
        large = []
        for tx in txs:
            total_out = sum(o.get("value", 0) for o in tx.get("out", []))
            if total_out > threshold:
                large.append(total_out / 1e8)
        return {
            "count": len(large),
            "volume": sum(large),
        }
    except Exception as e:
        logger.debug(f"Whale tx fetch error: {e}")
        return {"count": 0, "volume": 0.0}

async def fetch_stock_prices(session: aiohttp.ClientSession,
                              logger: logging.Logger) -> dict:
    global _stock_cache
    import asyncio
    loop = asyncio.get_running_loop()

    def _fetch():
        global _stock_cache
        try:
            import yfinance as yf
            sp = yf.Ticker("^GSPC")
            nq = yf.Ticker("^IXIC")
            sp_info = sp.fast_info
            nq_info = nq.fast_info

            sp_price = float(sp_info.last_price)
            nq_price = float(nq_info.last_price)
            sp_prev = float(sp_info.previous_close)
            nq_prev = float(nq_info.previous_close)

            sp_ret = 0.0
            nq_ret = 0.0
            if _stock_cache["sp500"] > 0:
                sp_ret = (sp_price - _stock_cache["sp500"]) / _stock_cache["sp500"] * 100
            if _stock_cache["nasdaq"] > 0:
                nq_ret = (nq_price - _stock_cache["nasdaq"]) / _stock_cache["nasdaq"] * 100

            _stock_cache = {
                "sp500": sp_price, "nasdaq": nq_price,
                "sp500_ret": sp_ret, "nasdaq_ret": nq_ret,
                "ts": __import__("time").time(),
            }
            return _stock_cache
        except Exception as e:
            logger.debug(f"Stock fetch error: {e}")
            return _stock_cache

    return await loop.run_in_executor(None, _fetch)

async def fetch_onchain_data(session: aiohttp.ClientSession,
                              logger: logging.Logger) -> dict:
    result = {"mempool": 0, "active_addr": 0, "tx_volume": 0.0,
              "realized_cap": 0.0, "mempool_fee": 0.0}
    try:
        async with session.get(REST_MEMPOOL, timeout=aiohttp.ClientTimeout(total=8)) as r:
            text = await r.text()
            result["mempool"] = int(text.strip())
    except Exception:
        pass
    try:
        async with session.get(REST_BTC_STATS, timeout=aiohttp.ClientTimeout(total=8)) as r:
            data = await r.json(content_type=None)
            result["active_addr"] = int(data.get("n_unique_addresses", 0))
            result["tx_volume"] = float(data.get("estimated_btc_sent", 0)) / 1e8
            result["realized_cap"] = float(data.get("market_cap", 0)) / 1e9
            cost = float(data.get("cost_per_transaction", 0))
            result["mempool_fee"] = cost / 100
    except Exception as e:
        logger.debug(f"On-chain stats error: {e}")
    return result

async def fetch_gas_fees(session: aiohttp.ClientSession,
                          logger: logging.Logger) -> float:
    try:
        async with session.get(
            REST_GAS_FEES,
            timeout=aiohttp.ClientTimeout(total=8)
        ) as r:
            data = await r.json()
        result = data.get("result", {})
        return float(result.get("SafeGasPrice", 0))
    except Exception as e:
        logger.debug(f"Gas fees fetch error: {e}")
        return 0.0

async def fetch_spot_volume_24h(session: aiohttp.ClientSession,
                                symbol: str) -> float:
    try:
        url = REST_SPOT_24H.format(symbol=symbol)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            data = await r.json()
            return float(data.get("quoteVolume", 0))
    except Exception:
        return 0.0

def compute_cascade_risk(snapshot: dict, db_path: str, prefix: str,
                         logger: logging.Logger) -> dict:
    result = {"leverage_ratio": 0.0, "oi_velocity_1h": 0.0,
              "cascade_risk_score": 0.0, "cascade_direction": "neutral"}
    try:
        price = snapshot.get(f"{prefix}_price") or 0
        oi = snapshot.get(f"{prefix}_oi") or 0
        spot_vol = snapshot.get(f"{prefix}_spot_vol_24h") or 0
        ls_ratio = snapshot.get(f"{prefix}_ls_ratio")
        funding = snapshot.get(f"{prefix}_funding") or 0
        taker_buy = snapshot.get(f"{prefix}_taker_buy_vol") or 0
        taker_sell = snapshot.get(f"{prefix}_taker_sell_vol") or 0
        gex = snapshot.get(f"{prefix}_gex") or 0

        if oi <= 0 or price <= 0:
            logger.debug(f"[{prefix}] cascade: skip (no OI or price)")
            return result

        oi_usd = oi * price
        if spot_vol > 0:
            result["leverage_ratio"] = oi_usd / spot_vol

        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                f"SELECT {prefix}_oi FROM intraday_snapshots "
                f"WHERE {prefix}_oi IS NOT NULL AND {prefix}_oi > 0 "
                f"ORDER BY timestamp DESC LIMIT 1 OFFSET 12"
            ).fetchone()
            conn.close()
            if row and row[0] and row[0] > 0:
                result["oi_velocity_1h"] = (oi - row[0]) / row[0]
        except Exception:
            pass

        oi_v = abs(result["oi_velocity_1h"])
        oi_overload = min(oi_v / 0.015, 1.0)

        if ls_ratio is not None:
            ls_dev = abs(ls_ratio - 1.0)
            ls_imbalance = min(ls_dev / 1.06, 1.0)
        else:
            ls_imbalance = 0

        funding_norm = min(abs(funding * 100) / 0.0084, 1.0)

        lev = result["leverage_ratio"]
        leverage_norm = min(lev / 10.0, 1.0) if lev > 0 else 0

        total_taker = taker_buy + taker_sell
        if total_taker > 0:
            taker_imb_raw = abs((taker_buy - taker_sell) / (total_taker + 1e-10))
            taker_imb = min(taker_imb_raw / 0.42, 1.0)
        else:
            taker_imb = 0

        gex_regime = 1.0 if gex < 0 else 0.0

        score = (
            0.25 * oi_overload +
            0.20 * ls_imbalance +
            0.15 * funding_norm +
            0.20 * leverage_norm +
            0.10 * taker_imb +
            0.10 * gex_regime
        )
        result["cascade_risk_score"] = round(score, 4)

        if ls_ratio is not None:
            if ls_ratio > 1.5 and funding > 0:
                result["cascade_direction"] = "long_liq"
            elif ls_ratio < 0.67 and funding < 0:
                result["cascade_direction"] = "short_squeeze"

    except Exception as e:
        logger.debug(f"Cascade risk compute error: {e}")

    return result

async def _run_liquidation_stream(logger: logging.Logger):
    global _liq_buffer, _liq_buffer_lock
    import time as _time
    backoff = 1
    while _running:
        try:
            async with aiohttp.ClientSession() as ws_session:
                async with ws_session.ws_connect(
                    WS_LIQUIDATIONS,
                    heartbeat=30,
                    timeout=aiohttp.ClientWSTimeout(ws_receive=60),
                ) as ws:
                    logger.info("Liquidation WS: подключён к !forceOrder@arr")
                    backoff = 1
                    async for msg in ws:
                        if not _running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                import json as _json
                                data = _json.loads(msg.data)
                                o = data.get("o", {})
                                symbol = o.get("s", "")
                                if symbol not in ("BTCUSDT", "ETHUSDT"):
                                    continue
                                side   = o.get("S", "")
                                qty    = float(o.get("q", 0))
                                price  = float(o.get("ap", 0))
                                ts_ms  = int(o.get("T", _time.time() * 1000))
                                usd    = qty * price
                                if usd <= 0:
                                    continue
                                async with _liq_buffer_lock:
                                    _liq_buffer.append({
                                        "symbol": symbol,
                                        "side": side,
                                        "usd": usd,
                                        "ts_ms": ts_ms,
                                    })
                            except Exception:
                                pass
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
        except Exception as e:
            logger.warning(f"Liquidation WS отключён: {e}, reconnect через {backoff}s")
        if not _running:
            break
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)

async def get_liquidations_5m(symbol: str) -> dict:
    global _liq_buffer, _liq_buffer_lock
    import time as _time
    default = {"long_liq_usd": 0.0, "short_liq_usd": 0.0, "liq_imb": 0.0, "liq_total_usd": 0.0}
    if _liq_buffer_lock is None:
        return default
    try:
        cutoff_ms = int(_time.time() * 1000) - 5 * 60 * 1000
        async with _liq_buffer_lock:
            _liq_buffer[:] = [e for e in _liq_buffer if e["ts_ms"] >= cutoff_ms]
            relevant = [e for e in _liq_buffer if e["symbol"] == symbol]

        long_liq  = sum(e["usd"] for e in relevant if e["side"] == "SELL")
        short_liq = sum(e["usd"] for e in relevant if e["side"] == "BUY")
        total = long_liq + short_liq
        imb = (long_liq - short_liq) / total if total > 0 else 0.0
        return {"long_liq_usd": long_liq, "short_liq_usd": short_liq,
                "liq_imb": imb, "liq_total_usd": total}
    except Exception:
        return default

async def fetch_large_trades(session: aiohttp.ClientSession, symbol: str,
                              min_usd: float = 50_000) -> dict:
    default = {"large_buy_usd": 0.0, "large_sell_usd": 0.0,
               "large_trade_imb": 0.0, "large_trade_count": 0}
    try:
        import time as _time
        since_ms = int(_time.time() * 1000) - 5 * 60 * 1000
        end_ms   = int(_time.time() * 1000)
        buy_usd, sell_usd, count = 0.0, 0.0, 0
        from_id = None

        for _ in range(3):
            if from_id is not None:
                url = (f"{REST_AGG_TRADES.format(symbol=symbol)}"
                       f"&startTime={since_ms}&endTime={end_ms}&fromId={from_id}&limit=1000")
            else:
                url = (f"{REST_AGG_TRADES.format(symbol=symbol)}"
                       f"&startTime={since_ms}&endTime={end_ms}&limit=1000")
            data = await _retry_get(session, url, timeout=5.0)
            if not data or not isinstance(data, list):
                break
            for t in data:
                price = float(t.get("p", 0))
                qty   = float(t.get("q", 0))
                usd   = price * qty
                if usd < min_usd:
                    continue
                count += 1
                if t.get("m"):
                    sell_usd += usd
                else:
                    buy_usd += usd
            if len(data) < 1000:
                break
            from_id = int(data[-1]["a"]) + 1

        total = buy_usd + sell_usd
        imb = (buy_usd - sell_usd) / total if total > 0 else 0.0
        return {"large_buy_usd": buy_usd, "large_sell_usd": sell_usd,
                "large_trade_imb": imb, "large_trade_count": count}
    except Exception:
        return default

async def collect_snapshot(session: aiohttp.ClientSession,
                            db_path: str, whale_db: str,
                            logger: logging.Logger,
                            sentiment_cache: dict = None):
    ts = datetime.now(MOSCOW_TZ).strftime("%Y-%m-%d %H:%M")

    tasks = []
    for sym in SYMBOLS:
        tasks += [
            fetch_price(session, sym),
            fetch_oi(session, sym),
            fetch_orderbook(session, sym),
            fetch_volume_5m(session, sym),
            fetch_funding_rate(session, sym),
            fetch_taker_volume(session, sym),
            fetch_ls_ratio(session, sym),
            fetch_orderbook_depth(session, sym),
            fetch_deribit_options(session,
                "BTC" if "BTC" in sym else "ETH", logger),
        ]
    tasks.append(fetch_stablecoin_flows(session, logger))
    tasks.append(fetch_sol_price(session))
    tasks.append(fetch_bybit_price(session, "BTCUSDT"))
    tasks.append(fetch_bybit_price(session, "ETHUSDT"))
    tasks.append(fetch_gas_fees(session, logger))
    tasks.append(fetch_alt_prices(session))
    tasks.append(fetch_btc_dominance(session, logger))
    tasks.append(fetch_whale_transactions(session, logger))
    tasks.append(fetch_stock_prices(session, logger))
    tasks.append(fetch_onchain_data(session, logger))
    tasks.append(fetch_spot_volume_24h(session, "BTCUSDT"))
    tasks.append(fetch_spot_volume_24h(session, "ETHUSDT"))
    tasks.append(fetch_large_trades(session, "BTCUSDT"))
    tasks.append(fetch_large_trades(session, "ETHUSDT"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    btc_liq = await get_liquidations_5m("BTCUSDT")
    eth_liq = await get_liquidations_5m("ETHUSDT")

    def safe(val, default=0.0):
        return val if not isinstance(val, Exception) and val else default

    def safe_none(val):
        if isinstance(val, Exception) or val is None:
            return None
        return val

    def safe_dict(val):
        return val if isinstance(val, dict) else {}

    PER_SYM = 9

    btc_price    = safe_none(results[0])
    btc_oi       = safe_none(results[1])
    btc_ob       = safe_dict(results[2])
    btc_vol5m    = safe(results[3])
    btc_funding  = safe(results[4])
    btc_taker    = safe_dict(results[5])
    btc_ls       = safe(results[6])
    btc_depth    = safe_dict(results[7])
    btc_deribit  = safe_dict(results[8])

    eth_price    = safe_none(results[PER_SYM + 0])
    eth_oi       = safe_none(results[PER_SYM + 1])
    eth_ob       = safe_dict(results[PER_SYM + 2])
    eth_vol5m    = safe(results[PER_SYM + 3])
    eth_funding  = safe(results[PER_SYM + 4])
    eth_taker    = safe_dict(results[PER_SYM + 5])
    eth_ls       = safe(results[PER_SYM + 6])
    eth_depth    = safe_dict(results[PER_SYM + 7])
    eth_deribit  = safe_dict(results[PER_SYM + 8])

    stable = safe_dict(results[2 * PER_SYM])
    if not stable:
        stable = _stablecoin_cache

    sol_price       = safe(results[2 * PER_SYM + 1])
    btc_bybit_price = safe(results[2 * PER_SYM + 2])
    eth_bybit_price = safe(results[2 * PER_SYM + 3])
    gas_fee         = safe(results[2 * PER_SYM + 4])
    alt_prices      = safe_dict(results[2 * PER_SYM + 5])
    btc_dominance   = safe(results[2 * PER_SYM + 6])
    whale_tx        = safe_dict(results[2 * PER_SYM + 7])
    stocks          = safe_dict(results[2 * PER_SYM + 8])
    onchain         = safe_dict(results[2 * PER_SYM + 9])
    btc_spot_vol_24h = safe(results[2 * PER_SYM + 10])
    eth_spot_vol_24h = safe(results[2 * PER_SYM + 11])
    btc_large        = safe_dict(results[2 * PER_SYM + 12])
    eth_large        = safe_dict(results[2 * PER_SYM + 13])

    whale_flow = get_whale_flow_5m(whale_db)

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR IGNORE INTO intraday_snapshots (
                timestamp, btc_price, eth_price,
                btc_oi, eth_oi,
                btc_ob_imb, eth_ob_imb,
                btc_spread_pct, eth_spread_pct,
                btc_volume_5m, eth_volume_5m,
                btc_whale_buy, btc_whale_sell,
                eth_whale_buy, eth_whale_sell,
                btc_funding, eth_funding,
                usdt_circ, usdc_circ,
                usdt_delta_24h, usdc_delta_24h,
                btc_taker_buy_vol, btc_taker_sell_vol,
                eth_taker_buy_vol, eth_taker_sell_vol,
                btc_ls_ratio, eth_ls_ratio,
                btc_depth_imb_5, btc_depth_imb_20,
                eth_depth_imb_5, eth_depth_imb_20,
                btc_pc_vol_ratio, btc_pc_oi_ratio,
                eth_pc_vol_ratio, eth_pc_oi_ratio,
                sol_price, btc_bybit_price, eth_bybit_price, gas_fee,
                btc_gex, eth_gex,
                btc_options_skew, eth_options_skew,
                btc_max_pain, eth_max_pain,
                bnb_price, xrp_price, link_price, btc_dominance,
                social_sentiment, social_volume, trends_interest,
                whale_tx_count, whale_tx_volume,
                sp500_price, nasdaq_price, sp500_ret_5m, nasdaq_ret_5m,
                mempool_size, active_addresses, btc_tx_volume,
                btc_wmid_dev, eth_wmid_dev,
                btc_price_impact, eth_price_impact,
                btc_queue_imb, eth_queue_imb,
                realized_cap, mempool_fee,
                btc_long_liq_usd, btc_short_liq_usd, btc_liq_imb, btc_liq_total_usd,
                eth_long_liq_usd, eth_short_liq_usd, eth_liq_imb, eth_liq_total_usd,
                btc_large_buy_usd, btc_large_sell_usd, btc_large_trade_imb, btc_large_trade_count,
                eth_large_buy_usd, eth_large_sell_usd, eth_large_trade_imb, eth_large_trade_count
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts,
            btc_price or None, eth_price or None,
            btc_oi or None, eth_oi or None,
            btc_ob.get("imbalance"), eth_ob.get("imbalance"),
            btc_ob.get("spread_pct"), eth_ob.get("spread_pct"),
            btc_vol5m or None, eth_vol5m or None,
            whale_flow["BTCUSDT"]["buy"], whale_flow["BTCUSDT"]["sell"],
            whale_flow["ETHUSDT"]["buy"], whale_flow["ETHUSDT"]["sell"],
            btc_funding or None, eth_funding or None,
            stable.get("usdt_circ") or None,
            stable.get("usdc_circ") or None,
            stable.get("usdt_delta_24h") or None,
            stable.get("usdc_delta_24h") or None,
            btc_taker.get("buy_vol") or None,
            btc_taker.get("sell_vol") or None,
            eth_taker.get("buy_vol") or None,
            eth_taker.get("sell_vol") or None,
            btc_ls or None, eth_ls or None,
            btc_depth.get("depth_imb_5"), btc_depth.get("depth_imb_20"),
            eth_depth.get("depth_imb_5"), eth_depth.get("depth_imb_20"),
            btc_deribit.get("pc_vol_ratio"), btc_deribit.get("pc_oi_ratio"),
            eth_deribit.get("pc_vol_ratio"), eth_deribit.get("pc_oi_ratio"),
            sol_price or None,
            btc_bybit_price or None,
            eth_bybit_price or None,
            gas_fee or None,
            btc_deribit.get("gex"), eth_deribit.get("gex"),
            btc_deribit.get("skew"), eth_deribit.get("skew"),
            btc_deribit.get("max_pain"), eth_deribit.get("max_pain"),
            alt_prices.get("bnb") or None,
            alt_prices.get("xrp") or None,
            alt_prices.get("link") or None,
            btc_dominance or None,
            sentiment_cache.get("social_sentiment") if sentiment_cache else None,
            sentiment_cache.get("social_volume") if sentiment_cache else None,
            sentiment_cache.get("trends_interest") if sentiment_cache else None,
            whale_tx.get("count") or None,
            whale_tx.get("volume") or None,
            stocks.get("sp500") or None,
            stocks.get("nasdaq") or None,
            stocks.get("sp500_ret") or None,
            stocks.get("nasdaq_ret") or None,
            onchain.get("mempool") or None,
            onchain.get("active_addr") or None,
            onchain.get("tx_volume") or None,
            btc_depth.get("wmid_dev"), eth_depth.get("wmid_dev"),
            btc_depth.get("price_impact"), eth_depth.get("price_impact"),
            btc_depth.get("queue_imb"), eth_depth.get("queue_imb"),
            onchain.get("realized_cap") or None,
            onchain.get("mempool_fee") or None,
            btc_liq.get("long_liq_usd") or None,
            btc_liq.get("short_liq_usd") or None,
            btc_liq.get("liq_imb") or None,
            btc_liq.get("liq_total_usd") or None,
            eth_liq.get("long_liq_usd") or None,
            eth_liq.get("short_liq_usd") or None,
            eth_liq.get("liq_imb") or None,
            eth_liq.get("liq_total_usd") or None,
            btc_large.get("large_buy_usd") or None,
            btc_large.get("large_sell_usd") or None,
            btc_large.get("large_trade_imb") or None,
            btc_large.get("large_trade_count") or None,
            eth_large.get("large_buy_usd") or None,
            eth_large.get("large_sell_usd") or None,
            eth_large.get("large_trade_imb") or None,
            eth_large.get("large_trade_count") or None,
        ))
        conn.commit()

        try:
            prev_funding = conn.execute(
                "SELECT btc_funding, eth_funding FROM intraday_snapshots "
                "WHERE timestamp < ? AND btc_funding IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 1", (ts,)
            ).fetchone()
            if prev_funding and btc_funding and eth_funding:
                btc_fd = btc_funding - prev_funding[0]
                eth_fd = eth_funding - prev_funding[1]
                conn.execute(
                    "UPDATE intraday_snapshots SET btc_funding_delta=?, eth_funding_delta=? WHERE timestamp=?",
                    (btc_fd, eth_fd, ts)
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Funding delta error: {e}")

        if _vol_estimators and btc_price and eth_price:
            try:
                btc_ret = None
                eth_ret = None
                prev = conn.execute(
                    "SELECT btc_price, eth_price FROM intraday_snapshots "
                    "WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1", (ts,)
                ).fetchone()
                if prev and prev[0] and prev[1]:
                    btc_ret = math.log(btc_price / prev[0])
                    eth_ret = math.log(eth_price / prev[1])

                btc_vol, eth_vol = None, None
                btc_anom, eth_anom = 0, 0
                if btc_ret is not None:
                    btc_vol = _vol_estimators["btc"].update(btc_ret)
                    btc_anom = 1 if _vol_estimators["btc"].is_anomaly() else 0
                if eth_ret is not None:
                    eth_vol = _vol_estimators["eth"].update(eth_ret)
                    eth_anom = 1 if _vol_estimators["eth"].is_anomaly() else 0

                conn.execute(
                    "UPDATE intraday_snapshots SET "
                    "btc_egarch_vol=?, eth_egarch_vol=?, "
                    "btc_vol_anomaly=?, eth_vol_anomaly=? "
                    "WHERE timestamp=?",
                    (btc_vol, eth_vol, btc_anom, eth_anom, ts)
                )
                conn.commit()
            except Exception as e:
                logger.debug(f"EGARCH update error: {e}")

        try:
            for pfx, spot_vol in [("btc", btc_spot_vol_24h), ("eth", eth_spot_vol_24h)]:
                snap_data = {
                    f"{pfx}_price": btc_price if pfx == "btc" else eth_price,
                    f"{pfx}_oi": btc_oi if pfx == "btc" else eth_oi,
                    f"{pfx}_spot_vol_24h": spot_vol,
                    f"{pfx}_ls_ratio": btc_ls if pfx == "btc" else eth_ls,
                    f"{pfx}_funding": btc_funding if pfx == "btc" else eth_funding,
                    f"{pfx}_taker_buy_vol": btc_taker.get("buy_vol") if pfx == "btc" else eth_taker.get("buy_vol"),
                    f"{pfx}_taker_sell_vol": btc_taker.get("sell_vol") if pfx == "btc" else eth_taker.get("sell_vol"),
                    f"{pfx}_gex": btc_deribit.get("gex") if pfx == "btc" else eth_deribit.get("gex"),
                }
                cascade = compute_cascade_risk(snap_data, db_path, pfx, logger)
                conn.execute(
                    f"UPDATE intraday_snapshots SET "
                    f"{pfx}_spot_vol_24h=?, {pfx}_leverage_ratio=?, "
                    f"{pfx}_oi_velocity_1h=?, {pfx}_cascade_risk=? "
                    f"WHERE timestamp=?",
                    (spot_vol, cascade["leverage_ratio"],
                     cascade["oi_velocity_1h"], cascade["cascade_risk_score"], ts)
                )
            conn.commit()
        except Exception as e:
            logger.debug(f"Cascade risk update error: {e}")

        conn.close()
        btc_str = f"${btc_price:,.0f}" if btc_price else "N/A"
        eth_str = f"${eth_price:,.0f}" if eth_price else "N/A"
        warn = " [!]" if (btc_price is None or eth_price is None) else ""
        logger.info(
            f"Intraday [{ts}]: BTC={btc_str}, ETH={eth_str}, "
            f"BTC OB imb={btc_ob.get('imbalance', 0):.2f}{warn}"
        )

        if btc_price is None:
            _send_alert("WARN", "btc_price_missing",
                        "⚠️ BTC price не удалось получить — Binance API issue")
        if eth_price is None:
            _send_alert("WARN", "eth_price_missing",
                        "⚠️ ETH price не удалось получить — Binance API issue")
        if btc_oi is None:
            _send_alert("WARN", "btc_oi_missing",
                        "⚠️ BTC open interest недоступен — проверьте API", cooldown_sec=1800)

        try:
            if btc_price:
                prev_row = sqlite3.connect(db_path).execute(
                    "SELECT btc_price FROM intraday_snapshots WHERE timestamp < ? "
                    "ORDER BY timestamp DESC LIMIT 1", (ts,)
                ).fetchone()
                if prev_row and prev_row[0] and prev_row[0] > 0:
                    change_pct = (btc_price - prev_row[0]) / prev_row[0] * 100
                    if abs(change_pct) >= 3.0:
                        direction = "📈" if change_pct > 0 else "📉"
                        _send_alert("CRITICAL", "btc_spike",
                                    f"{direction} <b>BTC:</b> {change_pct:+.2f}% за 5 минут!\n"
                                    f"Цена: ${btc_price:,.0f}",
                                    cooldown_sec=300)
            if eth_price:
                prev_row = sqlite3.connect(db_path).execute(
                    "SELECT eth_price FROM intraday_snapshots WHERE timestamp < ? "
                    "ORDER BY timestamp DESC LIMIT 1", (ts,)
                ).fetchone()
                if prev_row and prev_row[0] and prev_row[0] > 0:
                    change_pct = (eth_price - prev_row[0]) / prev_row[0] * 100
                    if abs(change_pct) >= 3.0:
                        direction = "📈" if change_pct > 0 else "📉"
                        _send_alert("CRITICAL", "eth_spike",
                                    f"{direction} <b>ETH:</b> {change_pct:+.2f}% за 5 минут!\n"
                                    f"Цена: ${eth_price:,.0f}",
                                    cooldown_sec=300)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Intraday save error: {e}")

async def start_intraday_collector(db_path: str, whale_db: str,
                                    logger: logging.Logger):
    global _running, _vol_estimators, _liq_buffer_lock
    _running = True
    _liq_buffer_lock = asyncio.Lock()
    init_intraday_db(db_path, logger)

    liq_task = asyncio.create_task(_run_liquidation_stream(logger))
    logger.info("Liquidation WebSocket stream запущен")

    _vol_estimators = {}
    try:
        from volatility_estimator import VolatilityEstimator
        conn = sqlite3.connect(db_path)
        for asset, col in [("btc", "btc_ret_5m"), ("eth", "eth_ret_5m")]:
            rows = conn.execute(
                f"SELECT {col} FROM intraday_snapshots "
                f"WHERE {col} IS NOT NULL ORDER BY timestamp"
            ).fetchall()
            est = VolatilityEstimator(asset=asset.upper())
            if rows:
                returns = [r[0] for r in rows if r[0] is not None]
                est.fit(returns)
                logger.info(f"EGARCH [{asset.upper()}]: обучен на {len(returns)} returns")
            _vol_estimators[asset] = est
        conn.close()
    except Exception as e:
        logger.warning(f"EGARCH init error (fallback на rolling std): {e}")
        _vol_estimators = {}

    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        logger.info("Intraday collector запущен (интервал 5 минут)")

        try:
            from collectors.social_sentiment import get_social_sentiment
        except ImportError:
            get_social_sentiment = None

        def _get_sentiment():
            if get_social_sentiment is None:
                return {}
            try:
                return get_social_sentiment(logger) or {}
            except Exception:
                return {}

        sent = _get_sentiment()
        await collect_snapshot(session, db_path, whale_db, logger, sentiment_cache=sent)

        tick = 0
        while _running:
            await asyncio.sleep(INTERVAL_SEC)
            if not _running:
                break
            sent = _get_sentiment()
            await collect_snapshot(session, db_path, whale_db, logger, sentiment_cache=sent)
            backfill_returns(db_path, logger)
            tick += 1
            if tick % 288 == 0:
                cleanup_intraday(db_path, logger)

def stop_intraday_collector():
    global _running
    _running = False

def get_intraday_stats(db_path: str, minutes: int = 30) -> dict:
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M")
    result = {}
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT timestamp, btc_price, eth_price,
                   btc_ob_imb, eth_ob_imb,
                   btc_whale_buy, btc_whale_sell,
                   eth_whale_buy, eth_whale_sell
            FROM intraday_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp
        """, (cutoff,)).fetchall()
        conn.close()

        if not rows:
            return {}

        btc_prices = [r[1] for r in rows if r[1]]
        eth_prices = [r[2] for r in rows if r[2]]
        btc_imb    = [r[3] for r in rows if r[3] is not None]
        eth_imb    = [r[4] for r in rows if r[4] is not None]

        result = {
            "snapshots": len(rows),
            "btc": {
                "price_start": btc_prices[0] if btc_prices else None,
                "price_end":   btc_prices[-1] if btc_prices else None,
                "change_pct":  (btc_prices[-1] / btc_prices[0] - 1) * 100
                               if len(btc_prices) >= 2 else 0,
                "avg_imbalance": sum(btc_imb) / len(btc_imb) if btc_imb else 0.5,
                "whale_buy":  sum(r[5] or 0 for r in rows),
                "whale_sell": sum(r[6] or 0 for r in rows),
            },
            "eth": {
                "price_start": eth_prices[0] if eth_prices else None,
                "price_end":   eth_prices[-1] if eth_prices else None,
                "change_pct":  (eth_prices[-1] / eth_prices[0] - 1) * 100
                               if len(eth_prices) >= 2 else 0,
                "avg_imbalance": sum(eth_imb) / len(eth_imb) if eth_imb else 0.5,
                "whale_buy":  sum(r[7] or 0 for r in rows),
                "whale_sell": sum(r[8] or 0 for r in rows),
            },
        }
    except Exception as e:
        result = {"error": str(e)}

    return result

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | intraday | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("intraday")
    asyncio.run(start_intraday_collector(
        "data/intraday.db", "data/whale.db", log
    ))
