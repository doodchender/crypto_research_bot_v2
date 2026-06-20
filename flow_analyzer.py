import asyncio
import json
import logging
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta

import aiohttp
import pytz

MOSCOW_TZ      = pytz.timezone("Europe/Moscow")
INTERVAL_SEC   = 60
RETENTION_DAYS = 30
SYMBOLS        = ["BTCUSDT", "ETHUSDT"]

WS_BINANCE_TMPL = "wss://stream.binance.com:9443/ws/{symbol}@aggTrade"
WS_BYBIT        = "wss://stream.bybit.com/v5/public/spot"
WS_OKX          = "wss://ws.okx.com:8443/ws/v5/public"

def init_flow_db(db_path: str, logger: logging.Logger):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flow_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            exchange        TEXT NOT NULL DEFAULT 'all',
            trade_count     INTEGER,
            trades_per_min  REAL,
            buy_count       INTEGER,
            sell_count      INTEGER,
            buy_ratio       REAL,
            total_volume    REAL,
            buy_volume      REAL,
            sell_volume     REAL,
            avg_trade_size  REAL,
            avg_buy_size    REAL,
            avg_sell_size   REAL,
            avg7_trade_count    REAL,
            avg7_trades_per_min REAL,
            avg7_avg_trade_size REAL,
            avg7_buy_ratio      REAL,
            count_zscore    REAL,
            size_zscore     REAL,
            signal          TEXT,
            UNIQUE(timestamp, symbol, exchange)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_flow_ts
        ON flow_snapshots(timestamp, symbol, exchange)
    """)
    conn.commit()
    conn.close()
    logger.info(f"Flow DB инициализирована: {db_path}")

def _cleanup(db_path: str, logger: logging.Logger):
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(db_path)
        n = conn.execute("DELETE FROM flow_snapshots WHERE timestamp < ?", (cutoff,)).rowcount
        conn.commit()
        conn.close()
        if n:
            logger.info(f"Flow cleanup: {n} записей")
    except Exception as e:
        logger.error(f"Flow cleanup: {e}")

def _aggregate(trades: list) -> dict:
    if not trades:
        return {}
    buys  = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]
    n  = len(trades)
    nb = len(buys)
    ns = len(sells)
    total_vol = sum(t["vol"] for t in trades)
    buy_vol   = sum(t["vol"] for t in buys)
    sell_vol  = sum(t["vol"] for t in sells)
    return {
        "trade_count":    n,
        "trades_per_min": round(n / (INTERVAL_SEC / 60), 1),
        "buy_count":      nb,
        "sell_count":     ns,
        "buy_ratio":      nb / n if n else 0.5,
        "total_volume":   total_vol,
        "buy_volume":     buy_vol,
        "sell_volume":    sell_vol,
        "avg_trade_size": total_vol / n if n else 0,
        "avg_buy_size":   buy_vol / nb if nb else 0,
        "avg_sell_size":  sell_vol / ns if ns else 0,
    }

def _get_7day_avg(db_path: str, symbol: str, exchange: str = "all") -> dict:
    cutoff = (datetime.now(MOSCOW_TZ) - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT trade_count, trades_per_min, avg_trade_size, buy_ratio
            FROM flow_snapshots
            WHERE symbol=? AND exchange=? AND timestamp>=? AND trade_count>0
        """, (symbol, exchange, cutoff)).fetchall()
        conn.close()
        if len(rows) < 3:
            return {}
        counts = [r[0] for r in rows]
        tpms   = [r[1] for r in rows]
        sizes  = [r[2] for r in rows]
        ratios = [r[3] for r in rows]
        def mean(x): return sum(x) / len(x)
        def std(x):
            m = mean(x)
            v = sum((i - m)**2 for i in x) / len(x)
            return v**0.5 if v > 0 else 1.0
        return {
            "avg7_trade_count":    mean(counts),
            "avg7_trades_per_min": mean(tpms),
            "avg7_avg_trade_size": mean(sizes),
            "avg7_buy_ratio":      mean(ratios),
            "std7_count":          std(counts),
            "std7_size":           std(sizes),
        }
    except Exception:
        return {}

def _get_signal(m: dict, a: dict) -> tuple:
    if not a:
        return "NORMAL", 0.0, 0.0
    sc = (m["trade_count"] - a["avg7_trade_count"]) / a["std7_count"] if a.get("std7_count") and a["std7_count"] > 0 else 0.0
    ss = (m["avg_trade_size"] - a["avg7_avg_trade_size"]) / a["std7_size"] if a.get("std7_size") and a["std7_size"] > 0 else 0.0
    br = m["buy_ratio"]
    if sc > 2.0 and br > 0.65:
        return "FOMO", sc, ss
    if sc > 2.0 and br < 0.35:
        return "PANIC", sc, ss
    if sc > 1.5 or ss > 1.5:
        return "ACTIVE", sc, ss
    return "NORMAL", sc, ss

def _save(db_path: str, symbol: str, exchange: str, m: dict,
          a: dict, logger: logging.Logger):
    ts = datetime.now(MOSCOW_TZ).strftime("%Y-%m-%d %H:%M")
    sig, sc, ss = _get_signal(m, a)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR REPLACE INTO flow_snapshots (
                timestamp, symbol, exchange,
                trade_count, trades_per_min, buy_count, sell_count,
                buy_ratio, total_volume, buy_volume, sell_volume,
                avg_trade_size, avg_buy_size, avg_sell_size,
                avg7_trade_count, avg7_trades_per_min,
                avg7_avg_trade_size, avg7_buy_ratio,
                count_zscore, size_zscore, signal
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts, symbol, exchange,
            m["trade_count"], m["trades_per_min"],
            m["buy_count"], m["sell_count"],
            m["buy_ratio"], m["total_volume"],
            m["buy_volume"], m["sell_volume"],
            m["avg_trade_size"], m["avg_buy_size"], m["avg_sell_size"],
            a.get("avg7_trade_count"), a.get("avg7_trades_per_min"),
            a.get("avg7_avg_trade_size"), a.get("avg7_buy_ratio"),
            sc, ss, sig
        ))
        conn.commit()
        conn.close()
        logger.info(
            f"Flow [{exchange}/{symbol}] {ts}: {m['trade_count']} сделок "
            f"{m['trades_per_min']}/мин avg=${m['avg_trade_size']:,.0f} "
            f"buy={m['buy_ratio']*100:.0f}% {sig}"
        )
    except Exception as e:
        logger.error(f"Flow save [{exchange}/{symbol}]: {e}")

async def _listen_binance(symbol: str, buf: dict, lock: asyncio.Lock,
                           logger: logging.Logger, session: aiohttp.ClientSession):
    url = WS_BINANCE_TMPL.format(symbol=symbol.lower())
    while True:
        try:
            async with session.ws_connect(url, heartbeat=30) as ws:
                logger.info(f"Flow Binance WS: {symbol}")
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    t     = json.loads(msg.data)
                    price = float(t.get("p", 0))
                    qty   = float(t.get("q", 0))
                    side  = "sell" if t.get("m") else "buy"
                    vol   = price * qty
                    async with lock:
                        buf[f"binance:{symbol}"].append({"side": side, "vol": vol})
                        buf[f"all:{symbol}"].append({"side": side, "vol": vol})
        except Exception as e:
            logger.warning(f"Flow Binance [{symbol}]: {e}, reconnect 5s")
            await asyncio.sleep(5)

async def _listen_bybit(buf: dict, lock: asyncio.Lock,
                         logger: logging.Logger, session: aiohttp.ClientSession):
    while True:
        try:
            async with session.ws_connect(WS_BYBIT, heartbeat=20) as ws:
                sub = {"op": "subscribe", "args": [f"publicTrade.{s}" for s in SYMBOLS]}
                await ws.send_str(json.dumps(sub))
                logger.info("Flow Bybit WS подключён")
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if not data.get("topic", "").startswith("publicTrade"):
                        continue
                    for t in data.get("data", []):
                        symbol = t.get("s", "")
                        if symbol not in SYMBOLS:
                            continue
                        price = float(t.get("p", 0))
                        qty   = float(t.get("v", 0))
                        side  = "sell" if t.get("S") == "Sell" else "buy"
                        vol   = price * qty
                        async with lock:
                            buf[f"bybit:{symbol}"].append({"side": side, "vol": vol})
                            buf[f"all:{symbol}"].append({"side": side, "vol": vol})
        except Exception as e:
            logger.warning(f"Flow Bybit: {e}, reconnect 5s")
            await asyncio.sleep(5)

async def _listen_okx(buf: dict, lock: asyncio.Lock,
                       logger: logging.Logger, session: aiohttp.ClientSession):
    while True:
        try:
            async with session.ws_connect(WS_OKX, heartbeat=25) as ws:
                args = [{"channel": "trades", "instId": s.replace("USDT", "-USDT")}
                        for s in SYMBOLS]
                await ws.send_str(json.dumps({"op": "subscribe", "args": args}))
                logger.info("Flow OKX WS подключён")
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data.get("arg", {}).get("channel") != "trades":
                        continue
                    for t in data.get("data", []):
                        inst   = t.get("instId", "")
                        symbol = inst.replace("-USDT", "USDT")
                        if symbol not in SYMBOLS:
                            continue
                        price = float(t.get("px", 0))
                        qty   = float(t.get("sz", 0))
                        side  = t.get("side", "buy")
                        vol   = price * qty
                        async with lock:
                            buf[f"okx:{symbol}"].append({"side": side, "vol": vol})
                            buf[f"all:{symbol}"].append({"side": side, "vol": vol})
        except Exception as e:
            logger.warning(f"Flow OKX: {e}, reconnect 5s")
            await asyncio.sleep(5)

async def _aggregator(buf: dict, lock: asyncio.Lock, db_path: str,
                       logger: logging.Logger):
    while True:
        await asyncio.sleep(INTERVAL_SEC)
        exchanges = ["binance", "bybit", "okx", "all"]
        for symbol in SYMBOLS:
            for exch in exchanges:
                key = f"{exch}:{symbol}"
                async with lock:
                    trades = list(buf[key])
                    buf[key].clear()
                if not trades:
                    continue
                m = _aggregate(trades)
                a = _get_7day_avg(db_path, symbol, exch)
                _save(db_path, symbol, exch, m, a, logger)
        _cleanup(db_path, logger)

def get_flow_summary(db_path: str, symbol: str, exchange: str = "all") -> dict:
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("""
            SELECT timestamp, trade_count, trades_per_min, buy_ratio,
                   avg_trade_size, total_volume, signal,
                   avg7_trade_count, avg7_avg_trade_size, count_zscore
            FROM flow_snapshots
            WHERE symbol=? AND exchange=?
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol, exchange)).fetchone()
        conn.close()
        if not row:
            return {}
        return {
            "timestamp":      row[0],
            "trade_count":    row[1],
            "trades_per_min": row[2],
            "buy_ratio":      row[3],
            "avg_trade_size": row[4],
            "total_volume":   row[5],
            "signal":         row[6],
            "avg7_count":     row[7],
            "avg7_size":      row[8],
            "count_zscore":   row[9],
        }
    except Exception:
        return {}

def get_flow_signal_text(db_path: str) -> str:
    lines = []
    for symbol in SYMBOLS:
        sym = symbol.replace("USDT", "")
        f = get_flow_summary(db_path, symbol)
        if not f:
            continue
        sig = f.get("signal", "NORMAL")
        tpm = f.get("trades_per_min", 0)
        buy = f.get("buy_ratio", 0.5) * 100
        lines.append(f"{sym}: {sig} ({tpm:.0f}/мин buy={buy:.0f}%)")
    return " | ".join(lines)

async def start_flow_analyzer(db_path: str, logger: logging.Logger):
    init_flow_db(db_path, logger)
    buf  = defaultdict(list)
    lock = asyncio.Lock()
    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        logger.info("Flow analyzer запущен (Binance + Bybit + OKX)")
        await asyncio.gather(
            _listen_binance("BTCUSDT", buf, lock, logger, session),
            _listen_binance("ETHUSDT", buf, lock, logger, session),
            _listen_bybit(buf, lock, logger, session),
            _listen_okx(buf, lock, logger, session),
            _aggregator(buf, lock, db_path, logger),
        )

def stop_flow_analyzer():
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | flow | %(levelname)s | %(message)s")
    log = logging.getLogger("flow")
    asyncio.run(start_flow_analyzer("data/flow.db", log))
