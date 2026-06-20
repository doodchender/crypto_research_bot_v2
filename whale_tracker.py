import asyncio
import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pytz

MOSCOW_TZ = pytz.timezone("Europe/Moscow")

SYMBOLS        = ["BTCUSDT", "ETHUSDT"]
WHALE_PCT      = 0.005
RETENTION_DAYS = 7
OB_INTERVAL    = 300
OB_DEPTH       = 20

WS_BINANCE = "wss://stream.binance.com:9443/stream?streams={streams}"
WS_BYBIT   = "wss://stream.bybit.com/v5/public/spot"
WS_OKX     = "wss://ws.okx.com:8443/ws/v5/public"

REST_BINANCE_STATS  = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
REST_BINANCE_DEPTH  = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth}"
REST_BYBIT_STATS    = "https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
REST_BYBIT_DEPTH    = "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit={depth}"
REST_OKX_STATS      = "https://www.okx.com/api/v5/market/ticker?instId={instId}"
REST_OKX_DEPTH      = "https://www.okx.com/api/v5/market/books?instId={instId}&sz={depth}"

WHALE_NAMES = [
    "Сатоши Накамото", "Большой Шорт",    "Тихий Дракон",
    "Мистер 1000x",    "Железные Руки",   "Граф Монте-Крипто",
    "Ночной Арбитр",   "Чёрный Лебедь",   "Серый Кардинал",
    "Повелитель Ликвидаций", "Рыночный Призрак", "Вечный Бык",
    "Сибирский Медведь", "Лондонский Туман",  "Биткоин Кит",
    "Дон Альтконов",   "Ямайский Шторм",  "Молчаливый Гигант",
    "Охотник за Дном", "Снайпер Стакана", "Тёмный Пул",
    "Хозяин Фьючерса", "Лунный Трейдер",  "Король Спреда",
]

_daily_volume:   dict             = {}
_running:        bool             = False
_whale_queue:    asyncio.Queue    = None
_whale_callback                   = None

def set_whale_callback(fn) -> None:
    global _whale_callback
    _whale_callback = fn

def init_whale_db(db_path: str, logger: logging.Logger):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS whale_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            side        TEXT NOT NULL,   -- 'buy' / 'sell'
            price       REAL NOT NULL,
            quantity    REAL NOT NULL,
            volume_usd  REAL NOT NULL,
            daily_vol   REAL,            -- дневной объём на момент сделки
            pct_of_day  REAL,            -- volume_usd / daily_vol * 100
            is_whale    INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            bid_price   REAL,
            ask_price   REAL,
            bid_volume  REAL,   -- суммарный объём топ-20 bid
            ask_volume  REAL,   -- суммарный объём топ-20 ask
            spread_pct  REAL,   -- спред в %
            imbalance   REAL    -- bid_vol / (bid_vol + ask_vol) — 0.5 = нейтрально
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_whale_ts ON whale_trades(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ob_ts ON orderbook_snapshots(timestamp)")
    conn.commit()
    conn.close()
    logger.info(f"Whale DB инициализирована: {db_path}")

def save_whale_trade(db_path: str, exchange: str, symbol: str, side: str,
                     price: float, qty: float, vol_usd: float,
                     daily_vol: float, logger: logging.Logger):
    pct      = (vol_usd / daily_vol * 100) if daily_vol > 0 else 0
    ts       = datetime.now(MOSCOW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    nickname = random.choice(WHALE_NAMES)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO whale_trades
            (timestamp, exchange, symbol, side, price, quantity, volume_usd, daily_vol, pct_of_day)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (ts, exchange, symbol, side, price, qty, vol_usd, daily_vol, pct))
        conn.commit()
        conn.close()
        logger.info(
            f"🐋 КИТ [{exchange}] {symbol} {side.upper()} "
            f"«{nickname}» ${vol_usd:,.0f} ({pct:.2f}% дневного объёма) @ ${price:,.2f}"
        )
    except Exception as e:
        logger.error(f"save_whale_trade error: {e}")
        return

    if _whale_queue is not None and _whale_callback is not None:
        try:
            _whale_queue.put_nowait({
                "exchange": exchange, "symbol": symbol, "side": side,
                "vol": vol_usd, "price": price, "pct": pct, "nickname": nickname,
            })
        except asyncio.QueueFull:
            logger.warning("Whale notification queue full, пропускаем")

def save_orderbook_snapshot(db_path: str, exchange: str, symbol: str,
                             bid_price: float, ask_price: float,
                             bid_vol: float, ask_vol: float,
                             logger: logging.Logger):
    ts = datetime.now(MOSCOW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    spread_pct  = (ask_price - bid_price) / bid_price * 100 if bid_price > 0 else 0
    total       = bid_vol + ask_vol
    imbalance   = bid_vol / total if total > 0 else 0.5
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
                     INSERT INTO orderbook_snapshots
                     (timestamp, exchange, symbol, bid_price, ask_price, bid_volume, ask_volume, spread_pct, imbalance)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (ts, exchange, symbol, bid_price, ask_price, bid_vol, ask_vol, spread_pct, imbalance))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"save_ob error: {e}")

def cleanup_old_records(db_path: str, logger: logging.Logger):
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(db_path)
        n1 = conn.execute("DELETE FROM whale_trades WHERE timestamp < ?", (cutoff,)).rowcount
        n2 = conn.execute("DELETE FROM orderbook_snapshots WHERE timestamp < ?", (cutoff,)).rowcount
        conn.commit()
        conn.close()
        if n1 or n2:
            logger.info(f"Whale DB cleanup: удалено {n1} trades, {n2} OB snapshots")
    except Exception as e:
        logger.error(f"cleanup error: {e}")

async def fetch_daily_volume(session: aiohttp.ClientSession, symbol: str,
                              exchange: str, logger: logging.Logger) -> float:
    try:
        if exchange == "binance":
            url = REST_BINANCE_STATS.format(symbol=symbol)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
                return float(data.get("quoteVolume", 0))

        elif exchange == "bybit":
            url = REST_BYBIT_STATS.format(symbol=symbol)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
                items = data.get("result", {}).get("list", [])
                if items:
                    return float(items[0].get("turnover24h", 0))

        elif exchange == "okx":
            inst_id = symbol.replace("USDT", "-USDT")
            url = REST_OKX_STATS.format(instId=inst_id)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
                items = data.get("data", [])
                if items:
                    return float(items[0].get("volCcy24h", 0)) * float(items[0].get("last", 1))
    except Exception as e:
        logger.warning(f"fetch_daily_volume [{exchange}/{symbol}]: {e}")
    return 0.0

async def fetch_orderbook(session: aiohttp.ClientSession, symbol: str,
                           exchange: str, db_path: str, logger: logging.Logger):
    try:
        if exchange == "binance":
            url = REST_BINANCE_DEPTH.format(symbol=symbol, depth=OB_DEPTH)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if not bids or not asks:
                return
            bid_price = float(bids[0][0])
            ask_price = float(asks[0][0])
            bid_vol   = sum(float(b[0]) * float(b[1]) for b in bids)
            ask_vol   = sum(float(a[0]) * float(a[1]) for a in asks)

        elif exchange == "bybit":
            url = REST_BYBIT_DEPTH.format(symbol=symbol, depth=OB_DEPTH)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
            ob   = data.get("result", {})
            bids = ob.get("b", [])
            asks = ob.get("a", [])
            if not bids or not asks:
                return
            bid_price = float(bids[0][0])
            ask_price = float(asks[0][0])
            bid_vol   = sum(float(b[0]) * float(b[1]) for b in bids)
            ask_vol   = sum(float(a[0]) * float(a[1]) for a in asks)

        elif exchange == "okx":
            inst_id = symbol.replace("USDT", "-USDT")
            url = REST_OKX_DEPTH.format(instId=inst_id, depth=OB_DEPTH)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
            ob   = data.get("data", [{}])[0]
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if not bids or not asks:
                return
            bid_price = float(bids[0][0])
            ask_price = float(asks[0][0])
            bid_vol   = sum(float(b[0]) * float(b[1]) for b in bids)
            ask_vol   = sum(float(a[0]) * float(a[1]) for a in asks)
        else:
            return

        save_orderbook_snapshot(db_path, exchange, symbol,
                                bid_price, ask_price, bid_vol, ask_vol, logger)

    except Exception as e:
        logger.warning(f"fetch_orderbook [{exchange}/{symbol}]: {e}")

async def listen_binance(symbols: list, db_path: str,
                          logger: logging.Logger, session: aiohttp.ClientSession):
    streams = "/".join(f"{s.lower()}@aggTrade" for s in symbols)
    url     = WS_BINANCE.format(streams=streams)

    while _running:
        try:
            async with session.ws_connect(url, heartbeat=30) as ws:
                logger.info("Binance WebSocket подключён")
                async for msg in ws:
                    if not _running:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data   = json.loads(msg.data)
                    trade  = data.get("data", data)
                    symbol = trade.get("s", "")
                    price  = float(trade.get("p", 0))
                    qty    = float(trade.get("q", 0))
                    is_sell= trade.get("m", False)
                    vol    = price * qty
                    side   = "sell" if is_sell else "buy"

                    key     = f"{symbol}:binance"
                    day_vol = _daily_volume.get(key, 0)

                    if day_vol > 0 and vol / day_vol >= WHALE_PCT:
                        save_whale_trade(db_path, "binance", symbol, side,
                                         price, qty, vol, day_vol, logger)
        except Exception as e:
            backoff = min(getattr(listen_binance, '_bo', 5) * 2, 120)
            listen_binance._bo = backoff
            logger.warning(f"Binance WS error: {e}, reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
        else:
            listen_binance._bo = 5

async def listen_bybit(symbols: list, db_path: str,
                        logger: logging.Logger, session: aiohttp.ClientSession):
    while _running:
        try:
            async with session.ws_connect(WS_BYBIT, heartbeat=20) as ws:
                sub = {"op": "subscribe", "args": [f"publicTrade.{s}" for s in symbols]}
                await ws.send_str(json.dumps(sub))
                logger.info("Bybit WebSocket подключён")

                async for msg in ws:
                    if not _running:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data.get("topic", "").startswith("publicTrade"):
                        for trade in data.get("data", []):
                            symbol = trade.get("s", "")
                            price  = float(trade.get("p", 0))
                            qty    = float(trade.get("v", 0))
                            side   = "sell" if trade.get("S") == "Sell" else "buy"
                            vol    = price * qty
                            key    = f"{symbol}:bybit"
                            day_vol = _daily_volume.get(key, 0)
                            if day_vol > 0 and vol / day_vol >= WHALE_PCT:
                                save_whale_trade(db_path, "bybit", symbol, side,
                                                 price, qty, vol, day_vol, logger)
        except Exception as e:
            backoff = min(getattr(listen_bybit, '_bo', 5) * 2, 120)
            listen_bybit._bo = backoff
            logger.warning(f"Bybit WS error: {e}, reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
        else:
            listen_bybit._bo = 5

async def listen_okx(symbols: list, db_path: str,
                      logger: logging.Logger, session: aiohttp.ClientSession):
    while _running:
        try:
            async with session.ws_connect(WS_OKX, heartbeat=25) as ws:
                args = [{"channel": "trades", "instId": s.replace("USDT", "-USDT")}
                        for s in symbols]
                sub = {"op": "subscribe", "args": args}
                await ws.send_str(json.dumps(sub))
                logger.info("OKX WebSocket подключён")

                async for msg in ws:
                    if not _running:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data.get("arg", {}).get("channel") == "trades":
                        for trade in data.get("data", []):
                            inst_id = trade.get("instId", "")
                            symbol  = inst_id.replace("-USDT", "USDT")
                            price   = float(trade.get("px", 0))
                            qty     = float(trade.get("sz", 0))
                            side    = trade.get("side", "buy")
                            vol     = price * qty
                            key     = f"{symbol}:okx"
                            day_vol = _daily_volume.get(key, 0)
                            if day_vol > 0 and vol / day_vol >= WHALE_PCT:
                                save_whale_trade(db_path, "okx", symbol, side,
                                                 price, qty, vol, day_vol, logger)
        except Exception as e:
            backoff = min(getattr(listen_okx, '_bo', 5) * 2, 120)
            listen_okx._bo = backoff
            logger.warning(f"OKX WS error: {e}, reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
        else:
            listen_okx._bo = 5

async def volume_updater(symbols: list, db_path: str,
                          logger: logging.Logger, session: aiohttp.ClientSession):
    vol_interval = 600
    ob_counter   = 0

    while _running:
        for symbol in symbols:
            for exchange in ("binance", "bybit", "okx"):
                vol = await fetch_daily_volume(session, symbol, exchange, logger)
                if vol > 0:
                    key = f"{symbol}:{exchange}"
                    _daily_volume[key] = vol
                    logger.debug(f"Daily vol [{exchange}/{symbol}]: ${vol:,.0f}")

        ob_counter += vol_interval
        if ob_counter >= OB_INTERVAL:
            ob_counter = 0
            for symbol in symbols:
                for exchange in ("binance", "bybit", "okx"):
                    await fetch_orderbook(session, symbol, exchange, db_path, logger)

        cleanup_old_records(db_path, logger)

        await asyncio.sleep(vol_interval)

def get_whale_summary(db_path: str, hours: int = 1) -> dict:
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    result = {}

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT symbol, side, SUM(volume_usd), COUNT(*),
                   GROUP_CONCAT(exchange || ':' || side || ':' || ROUND(volume_usd) || ':' || price, '|')
            FROM whale_trades
            WHERE timestamp >= ?
            GROUP BY symbol, side
        """, (cutoff,)).fetchall()
        conn.close()

        for symbol, side, vol, cnt, details in rows:
            if symbol not in result:
                result[symbol] = {"buy_vol": 0, "sell_vol": 0, "count": 0, "trades": []}
            result[symbol][f"{side}_vol"] += vol or 0
            result[symbol]["count"] += cnt or 0
            if details:
                for d in details.split("|"):
                    parts = d.split(":")
                    if len(parts) >= 4:
                        result[symbol]["trades"].append({
                            "exchange": parts[0], "side": parts[1],
                            "vol": float(parts[2]), "price": float(parts[3])
                        })
    except Exception:
        pass

    return result

def get_orderbook_pressure(db_path: str, symbol: str, minutes: int = 30) -> dict:
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("""
            SELECT AVG(imbalance), AVG(spread_pct), COUNT(*)
            FROM orderbook_snapshots
            WHERE symbol = ? AND timestamp >= ?
        """, (symbol, cutoff)).fetchone()
        conn.close()
        if row and row[2] > 0:
            return {"imbalance": row[0], "spread_pct": row[1], "snapshots": row[2]}
    except Exception:
        pass
    return {"imbalance": 0.5, "spread_pct": 0, "snapshots": 0}

async def _whale_notifier(logger: logging.Logger):
    while True:
        try:
            item = await _whale_queue.get()
            if _whale_callback:
                try:
                    await _whale_callback(**item)
                except Exception as e:
                    logger.error(f"Whale callback error: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Whale notifier error: {e}")

async def start_whale_tracker(db_path: str, logger: logging.Logger,
                               symbols: list = None, whale_pct: float = None):
    global _running, WHALE_PCT, _whale_queue
    _running     = True
    _whale_queue = asyncio.Queue(maxsize=200)
    if symbols:
        global SYMBOLS
        SYMBOLS = symbols
    if whale_pct:
        WHALE_PCT = whale_pct

    init_whale_db(db_path, logger)

    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        logger.info(f"Whale tracker запущен: {SYMBOLS}, порог={WHALE_PCT*100:.1f}%")
        await asyncio.gather(
            volume_updater(SYMBOLS, db_path, logger, session),
            listen_binance(SYMBOLS, db_path, logger, session),
            listen_bybit(SYMBOLS, db_path, logger, session),
            listen_okx(SYMBOLS, db_path, logger, session),
            _whale_notifier(logger),
        )

def stop_whale_tracker():
    global _running
    _running = False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | whale | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("whale")
    asyncio.run(start_whale_tracker("data/whale.db", log))
