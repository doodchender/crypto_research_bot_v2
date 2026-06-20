import json
import math
import sqlite3
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

DB_PATH      = Path("data/intraday.db")
HOURLY_PATH  = Path("data/hourly.db")
FLOW_PATH    = Path("data/flow.db")
OUT_FILE     = Path("data/intraday_coefficients.json")
RF_DIR       = Path("data/rf_models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | train | %(levelname)s | %(message)s",
)
log = logging.getLogger("train")

def load_data(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM intraday_snapshots
        WHERE btc_ret_5m IS NOT NULL
          AND eth_ret_5m IS NOT NULL
          AND btc_price IS NOT NULL
          AND eth_price IS NOT NULL
          AND btc_volume_5m IS NOT NULL
          AND btc_ob_imb IS NOT NULL
        ORDER BY timestamp
    """).fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    log.info(f"Загружено {len(data)} записей с заполненными returns")
    return data

def load_hourly_data(db_path: str) -> list[dict]:
    if not Path(db_path).exists():
        log.warning(f"hourly.db не найдена: {db_path}")
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, fear_greed_value, btc_atr_pct, eth_atr_pct,
               media_sentiment
        FROM hourly_snapshots
        ORDER BY timestamp
    """).fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    log.info(f"Загружено {len(data)} hourly записей")
    return data

def load_flow_data(db_path: str) -> list[dict]:
    if not Path(db_path).exists():
        log.warning(f"flow.db не найдена: {db_path}")
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT timestamp, symbol, buy_ratio, count_zscore,
                   size_zscore, signal, buy_volume, sell_volume
            FROM flow_snapshots
            WHERE exchange = 'all'
            ORDER BY timestamp
        """).fetchall()
        conn.close()
        data = [dict(r) for r in rows]
        log.info(f"Загружено {len(data)} flow записей (exchange=all)")
        return data
    except Exception as e:
        log.warning(f"Ошибка чтения flow.db: {e}")
        return []

def match_hourly(intraday_ts: str, hourly_data: list[dict]) -> dict:
    best = None
    for h in hourly_data:
        if h["timestamp"] <= intraday_ts:
            best = h
        else:
            break
    return best or {}

def match_flow(intraday_ts: str, flow_data: list[dict], symbol: str) -> dict:
    sym_map = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
    target_sym = sym_map.get(symbol, "")
    best = None
    for f in flow_data:
        if f["symbol"] == target_sym and f["timestamp"] <= intraday_ts:
            best = f
        elif f["timestamp"] > intraday_ts:
            break
    return best or {}

def compute_technical_indicators(data: list[dict], i: int, prefix: str) -> dict:
    result = {
        "ema_diff": 0.0, "macd": 0.0, "macd_signal": 0.0,
        "bb_position": 0.0, "obv_change": 0.0, "vwap_dev": 0.0,
        "roll_measure": 0.0, "roll_impact": 0.0,
        "hurst": 0.5, "macd_1h": 0.0, "macd_15m": 0.0,
        "ichimoku_conv": 0.0, "ichimoku_base": 0.0, "ichimoku_span_a": 0.0,
        "supertrend": 0.0, "keltner_pos": 0.0,
        "volume_poc_dist": 0.0, "price_efficiency": 0.0,
        "volume_poc_1h": 0.0, "rolling_sharpe_6": 0.0, "rolling_sharpe_12": 0.0,
        "volume_trend": 0.0, "price_pos_30m": 0.0, "high_low_ratio": 0.0,
        "mtf_momentum_strength": 0.0,
        "dir_persistence": 0.0,
        "price_vs_ema5": 0.0,
        "price_vs_ema20": 0.0,
        "price_vs_ema50": 0.0,
        "price_vs_ema200": 0.0,
        "momentum_squeeze": 1.0,
        "path_efficiency_30": 0.0,
        "path_efficiency_60": 0.0,
        "fisher_transform": 0.0,
        "tsi": 0.0,
        "vol_conf_momentum": 0.0,
        "momentum_zscore": 0.0,
    }

    window = min(i + 1, 60)
    prices = []
    volumes = []
    for j in range(max(0, i - window + 1), i + 1):
        p = data[j].get(f"{prefix}_price")
        v = data[j].get(f"{prefix}_volume_5m")
        if p and p > 0:
            prices.append(p)
            volumes.append(v if v and v > 0 else 0)

    if len(prices) < 10:
        return result

    def ema(vals, span):
        alpha = 2 / (span + 1)
        e = vals[0]
        for v in vals[1:]:
            e = alpha * v + (1 - alpha) * e
        return e

    ema9 = ema(prices, 9)
    ema21 = ema(prices, 21)
    cur_price = prices[-1]

    result["ema_diff"] = (ema9 - ema21) / cur_price * 100 if cur_price > 0 else 0.0

    if len(prices) >= 26:
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd_line = ema12 - ema26
        result["macd"] = macd_line / cur_price * 100
        result["macd_signal"] = result["ema_diff"]

    if len(prices) >= 20:
        sma20 = sum(prices[-20:]) / 20
        std20 = math.sqrt(sum((p - sma20) ** 2 for p in prices[-20:]) / 20)
        if std20 > 0:
            result["bb_position"] = (cur_price - sma20) / (2 * std20)

    if len(prices) >= 3 and len(volumes) >= 3:
        obv = 0.0
        obv_prev = 0.0
        for k in range(1, len(prices)):
            if prices[k] > prices[k - 1]:
                obv += volumes[k]
            elif prices[k] < prices[k - 1]:
                obv -= volumes[k]
            if k == len(prices) - 2:
                obv_prev = obv
        if obv_prev != 0:
            result["obv_change"] = (obv - obv_prev) / (abs(obv_prev) + 1)

    if sum(volumes) > 0:
        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
        result["vwap_dev"] = (cur_price - vwap) / vwap * 100 if vwap > 0 else 0.0

    if len(prices) >= 5:
        rets = [math.log(prices[k] / prices[k - 1]) for k in range(1, len(prices))]
        if len(rets) >= 4:
            n = len(rets)
            mean_r = sum(rets) / n
            cov = sum((rets[k] - mean_r) * (rets[k - 1] - mean_r)
                      for k in range(1, n)) / (n - 1)
            if cov < 0:
                result["roll_measure"] = 2 * math.sqrt(-cov)
            avg_vol = sum(volumes[-5:]) / 5 if volumes else 0
            result["roll_impact"] = result["roll_measure"] * math.log(avg_vol + 1)

    if len(prices) >= 30:
        try:
            log_rets = [math.log(prices[k] / prices[k - 1]) for k in range(1, len(prices))]
            n = len(log_rets)
            mean_r = sum(log_rets) / n
            dev = [r - mean_r for r in log_rets]
            cum_dev = [sum(dev[:k + 1]) for k in range(n)]
            R = max(cum_dev) - min(cum_dev)
            S = math.sqrt(sum((r - mean_r) ** 2 for r in log_rets) / n)
            if S > 0 and R > 0:
                result["hurst"] = math.log(R / S) / math.log(n / 2) if n > 2 else 0.5
                result["hurst"] = max(0.1, min(0.9, result["hurst"]))
        except (ValueError, ZeroDivisionError):
            pass

    if len(prices) >= 40:
        prices_15m = prices[::3]
        if len(prices_15m) >= 26:
            ema12_15 = ema(prices_15m, 12)
            ema26_15 = ema(prices_15m, 26)
            result["macd_15m"] = (ema12_15 - ema26_15) / cur_price * 100

    if len(prices) >= 60:
        prices_1h = prices[::12]
        if len(prices_1h) >= 5:
            if len(prices_1h) >= 26:
                ema12_1h = ema(prices_1h, 12)
                ema26_1h = ema(prices_1h, 26)
                result["macd_1h"] = (ema12_1h - ema26_1h) / cur_price * 100
            else:
                mid = len(prices_1h) // 2
                avg_recent = sum(prices_1h[mid:]) / (len(prices_1h) - mid)
                avg_old = sum(prices_1h[:mid]) / mid
                result["macd_1h"] = (avg_recent - avg_old) / cur_price * 100

    if len(prices) >= 52:
        p9 = prices[-9:]
        p26 = prices[-26:]
        p52 = prices[-52:]
        tenkan = (max(p9) + min(p9)) / 2
        kijun = (max(p26) + min(p26)) / 2
        span_a = (tenkan + kijun) / 2
        span_b = (max(p52) + min(p52)) / 2
        result["ichimoku_conv"] = (tenkan - cur_price) / cur_price * 100
        result["ichimoku_base"] = (kijun - cur_price) / cur_price * 100
        result["ichimoku_span_a"] = (span_a - span_b) / cur_price * 100

    if len(prices) >= 14:
        atr_vals = [abs(prices[k] - prices[k - 1]) for k in range(len(prices) - 14, len(prices))]
        atr = sum(atr_vals) / 14 if atr_vals else 0
        if atr > 0:
            upper_band = sum(prices[-14:]) / 14 + 2 * atr
            lower_band = sum(prices[-14:]) / 14 - 2 * atr
            if cur_price > upper_band:
                result["supertrend"] = 1.0
            elif cur_price < lower_band:
                result["supertrend"] = -1.0
            else:
                result["supertrend"] = (cur_price - (upper_band + lower_band) / 2) / atr

    if len(prices) >= 20:
        atr_vals = [abs(prices[k] - prices[k - 1]) for k in range(max(1, len(prices) - 20), len(prices))]
        atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0
        kc_mid = ema(prices[-20:], 20)
        if atr > 0:
            result["keltner_pos"] = (cur_price - kc_mid) / (2 * atr)
            result["keltner_pos"] = max(-2, min(2, result["keltner_pos"]))

    if len(prices) >= 20 and sum(volumes) > 0:
        min_p = min(prices)
        max_p = max(prices)
        if max_p > min_p:
            bucket_size = (max_p - min_p) / 10
            buckets = [0.0] * 10
            for p, v in zip(prices, volumes):
                idx = min(int((p - min_p) / bucket_size), 9)
                buckets[idx] += v
            poc_idx = buckets.index(max(buckets))
            poc_price = min_p + (poc_idx + 0.5) * bucket_size
            result["volume_poc_dist"] = (cur_price - poc_price) / cur_price * 100

    if len(prices) >= 10:
        net_change = abs(prices[-1] - prices[-10])
        abs_changes = sum(abs(prices[k] - prices[k - 1]) for k in range(len(prices) - 9, len(prices)))
        if abs_changes > 0:
            result["price_efficiency"] = net_change / abs_changes

    if len(prices) >= 12 and sum(volumes[-12:]) > 0:
        p12 = prices[-12:]
        v12 = volumes[-12:]
        min_p, max_p = min(p12), max(p12)
        if max_p > min_p:
            bucket_size = (max_p - min_p) / 8
            buckets = [0.0] * 8
            for p, v in zip(p12, v12):
                idx = min(int((p - min_p) / bucket_size), 7)
                buckets[idx] += v
            poc_idx = buckets.index(max(buckets))
            poc_price = min_p + (poc_idx + 0.5) * bucket_size
            result["volume_poc_1h"] = (cur_price - poc_price) / cur_price * 100

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        return m / std if std > 0 else 0.0

    if len(prices) >= 7:
        rets6 = [math.log(prices[k] / prices[k - 1]) for k in range(len(prices) - 6, len(prices))]
        result["rolling_sharpe_6"] = _sharpe(rets6)
    if len(prices) >= 13:
        rets12 = [math.log(prices[k] / prices[k - 1]) for k in range(len(prices) - 12, len(prices))]
        result["rolling_sharpe_12"] = _sharpe(rets12)

    if len(volumes) >= 6:
        v6 = volumes[-6:]
        first3 = sum(v6[:3]) / 3
        last3 = sum(v6[3:]) / 3
        if first3 > 0:
            result["volume_trend"] = (last3 - first3) / first3

    if len(prices) >= 6:
        p6 = prices[-6:]
        h30 = max(p6)
        l30 = min(p6)
        if h30 > l30:
            result["price_pos_30m"] = (cur_price - l30) / (h30 - l30)

    if len(prices) >= 20:
        p10 = prices[-10:]
        p20_10 = prices[-20:-10]
        range_recent = max(p10) - min(p10)
        range_old = max(p20_10) - min(p20_10)
        if range_old > 0:
            result["high_low_ratio"] = range_recent / range_old

    def _sgn(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)
    result["mtf_momentum_strength"] = (
        _sgn(result["macd"]) + _sgn(result["macd_15m"]) + _sgn(result["macd_1h"])
    )

    if len(prices) >= 7:
        signs = [_sgn(math.log(prices[k] / prices[k - 1])) for k in range(len(prices) - 6, len(prices))]
        result["dir_persistence"] = sum(signs) / 6.0

    if len(prices) >= 14:
        atr_vals = [abs(prices[k] - prices[k - 1]) for k in range(len(prices) - 14, len(prices))]
        atr = sum(atr_vals) / 14 if atr_vals else 0
    else:
        atr = 0

    if atr > 0:
        ema5 = ema(prices, 5)
        result["price_vs_ema5"] = (cur_price - ema5) / atr
        ema20_val = ema(prices, 20)
        result["price_vs_ema20"] = (cur_price - ema20_val) / atr
        if len(prices) >= 50:
            ema50 = ema(prices, 50)
            result["price_vs_ema50"] = (cur_price - ema50) / atr
        if len(prices) >= 60:
            ema200 = ema(prices, min(200, len(prices)))
            result["price_vs_ema200"] = (cur_price - ema200) / atr

    if len(prices) >= 20 and atr > 0:
        sma20 = sum(prices[-20:]) / 20
        std20 = math.sqrt(sum((p - sma20) ** 2 for p in prices[-20:]) / 20)
        bb_width = 4 * std20
        kc_width = 4 * atr
        if kc_width > 0:
            result["momentum_squeeze"] = bb_width / kc_width

    def _path_eff(arr):
        if len(arr) < 2:
            return 0.0
        net = abs(arr[-1] - arr[0])
        total = sum(abs(arr[k] - arr[k - 1]) for k in range(1, len(arr)))
        return net / total if total > 0 else 0.0

    if len(prices) >= 30:
        result["path_efficiency_30"] = _path_eff(prices[-30:])
    if len(prices) >= 60:
        result["path_efficiency_60"] = _path_eff(prices[-60:])

    if len(prices) >= 7:
        rets = [math.log(prices[k] / prices[k - 1]) for k in range(len(prices) - 6, len(prices))]
        mean_r = sum(rets) / len(rets)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / len(rets)) or 1e-10
        x = rets[-1] / (3 * std_r)
        x = max(-0.999, min(0.999, x))
        try:
            result["fisher_transform"] = 0.5 * math.log((1 + x) / (1 - x))
        except (ValueError, ZeroDivisionError):
            result["fisher_transform"] = 0.0

    if len(prices) >= 26:
        pc = [prices[k] - prices[k - 1] for k in range(1, len(prices))]
        abs_pc = [abs(x) for x in pc]
        ema1 = ema(pc, 13)
        ema2 = ema([abs(x) for x in pc], 13)
        if ema2 > 0:
            result["tsi"] = ema1 / ema2 * 100

    if len(prices) >= 13 and sum(volumes) > 0:
        avg_vol = sum(volumes[-12:]) / 12
        cur_vol = volumes[-1]
        if avg_vol > 0:
            last_ret = math.log(prices[-1] / prices[-2]) if prices[-2] > 0 else 0
            result["vol_conf_momentum"] = last_ret * math.log(cur_vol / avg_vol + 1)

    if len(prices) >= 20:
        lookback = min(len(prices) - 1, 60)
        hist_rets = [math.log(prices[k] / prices[k - 1]) for k in range(len(prices) - lookback, len(prices) - 1)]
        if hist_rets:
            mean_h = sum(hist_rets) / len(hist_rets)
            std_h = math.sqrt(sum((r - mean_h) ** 2 for r in hist_rets) / len(hist_rets)) or 1e-10
            last_ret = math.log(prices[-1] / prices[-2]) if prices[-2] > 0 else 0
            result["momentum_zscore"] = (last_ret - mean_h) / std_h

    return result

def compute_vpin(flow_window: list[dict], n_buckets: int = 6) -> float:
    if len(flow_window) < n_buckets:
        return 0.0
    total_imb = 0.0
    total_vol = 0.0
    for f in flow_window[-n_buckets:]:
        bv = f.get("buy_volume", 0) or 0
        sv = f.get("sell_volume", 0) or 0
        total_imb += abs(bv - sv)
        total_vol += bv + sv
    return total_imb / (total_vol + 1)

def compute_entropy(flow_window: list[dict], n: int = 10) -> float:
    if len(flow_window) < 3:
        return 1.0
    ratios = []
    for f in flow_window[-n:]:
        br = f.get("buy_ratio")
        if br is not None:
            ratios.append(br)
    if len(ratios) < 3:
        return 1.0
    bins = [0] * 5
    for r in ratios:
        idx = min(int(r * 5), 4)
        bins[idx] += 1
    total = sum(bins)
    entropy = 0.0
    for b in bins:
        if b > 0:
            p = b / total
            entropy -= p * math.log2(p)
    return entropy / math.log2(5) if entropy > 0 else 0.0

def compute_kyle_lambda(ret: float, volume: float) -> float:
    if volume and volume > 0:
        return abs(ret) / math.log(volume + 1)
    return 0.0

def compute_amihud(ret: float, dollar_volume: float) -> float:
    if dollar_volume and dollar_volume > 0:
        return abs(ret) * 1e6 / dollar_volume
    return 0.0

def compute_correlation_breakdown(data: list[dict], i: int,
                                    window: int = 12) -> float:
    if i < window:
        return 0.0
    btc_rets = []
    eth_rets = []
    for j in range(i - window, i):
        br = data[j].get("btc_ret_5m")
        er = data[j].get("eth_ret_5m")
        if br is not None and er is not None:
            btc_rets.append(br)
            eth_rets.append(er)
    if len(btc_rets) < 6:
        return 0.0
    n = len(btc_rets)
    mean_b = sum(btc_rets) / n
    mean_e = sum(eth_rets) / n
    cov = sum((btc_rets[k] - mean_b) * (eth_rets[k] - mean_e) for k in range(n)) / n
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in btc_rets) / n)
    std_e = math.sqrt(sum((x - mean_e) ** 2 for x in eth_rets) / n)
    if std_b > 0 and std_e > 0:
        return cov / (std_b * std_e)
    return 0.0

def compute_fractal_dimension(data: list[dict], i: int, prefix: str,
                                window: int = 12, kmax: int = 4) -> float:
    if i < window:
        return 1.5
    prices = []
    for j in range(i - window, i):
        p = data[j].get(f"{prefix}_price")
        if p is not None:
            prices.append(p)
    if len(prices) < 8:
        return 1.5

    N = len(prices)
    L = []
    ks = []
    for k in range(1, min(kmax + 1, N // 2)):
        lengths = []
        for m in range(1, k + 1):
            length = 0.0
            count = 0
            idx = m - 1
            while idx + k < N:
                length += abs(prices[idx + k] - prices[idx])
                idx += k
                count += 1
            if count > 0:
                norm = (N - 1) / (count * k * k)
                lengths.append(length * norm)
        if lengths:
            avg_L = sum(lengths) / len(lengths)
            if avg_L > 0:
                L.append(math.log(avg_L))
                ks.append(math.log(1.0 / k))

    if len(L) < 2:
        return 1.5

    n = len(L)
    mean_k = sum(ks) / n
    mean_L = sum(L) / n
    num = sum((ks[j] - mean_k) * (L[j] - mean_L) for j in range(n))
    den = sum((ks[j] - mean_k) ** 2 for j in range(n))
    if den > 0:
        fd = num / den
        return max(1.0, min(2.0, fd))
    return 1.5

def get_flow_window(flow_by_sym: dict, prefix: str, flow_idx: int,
                     ts: str, window: int = 10) -> list[dict]:
    sym_flows = flow_by_sym.get(prefix, [])
    end = flow_idx
    start = max(0, end - window)
    return sym_flows[start:end]

def compute_features(data: list[dict], hourly_data: list[dict],
                      flow_data: list[dict]) -> list[dict]:
    enriched = []

    cvd_acc = {"btc": 0.0, "eth": 0.0}
    flow_idx = {"btc": 0, "eth": 0}
    sym_map = {"btc": "BTCUSDT", "eth": "ETHUSDT"}

    flow_by_sym = {"btc": [], "eth": []}
    for f in flow_data:
        for prefix in ("btc", "eth"):
            if f["symbol"] == sym_map[prefix]:
                flow_by_sym[prefix].append(f)

    for i, row in enumerate(data):
        ts = row["timestamp"]

        hourly = match_hourly(ts, hourly_data)

        for prefix in ("btc", "eth"):
            idx = flow_idx[prefix]
            sym_flows = flow_by_sym[prefix]
            while idx < len(sym_flows) and sym_flows[idx]["timestamp"] <= ts:
                bv = sym_flows[idx].get("buy_volume", 0) or 0
                sv = sym_flows[idx].get("sell_volume", 0) or 0
                cvd_acc[prefix] += bv - sv
                idx += 1
            flow_idx[prefix] = idx

        for prefix in ("btc", "eth"):
            feat = {}

            ob_imb = row.get(f"{prefix}_ob_imb")
            feat["ob_imb"] = (ob_imb - 0.5) * 2 if ob_imb is not None else 0.0

            vol = row.get(f"{prefix}_volume_5m")
            feat["log_volume"] = math.log(vol) if vol and vol > 0 else 0.0

            oi_now = row.get(f"{prefix}_oi")
            if i > 0 and oi_now:
                oi_prev = data[i - 1].get(f"{prefix}_oi")
                if oi_prev and oi_prev > 0:
                    feat["oi_change_pct"] = (oi_now - oi_prev) / oi_prev * 100
                else:
                    feat["oi_change_pct"] = 0.0
            else:
                feat["oi_change_pct"] = 0.0

            wb = row.get(f"{prefix}_whale_buy", 0) or 0
            ws = row.get(f"{prefix}_whale_sell", 0) or 0
            total_whale = wb + ws
            feat["whale_imb"] = (wb - ws) / (total_whale + 1)

            for lag_n in range(1, 7):
                if i >= lag_n:
                    val = data[i - lag_n].get(f"{prefix}_ret_5m", 0) or 0
                else:
                    val = 0.0
                feat[f"ret_lag{lag_n}"] = val

            spread = row.get(f"{prefix}_spread_pct")
            feat["spread_pct"] = spread if spread is not None else 0.0

            if i >= 3:
                p_now = row.get(f"{prefix}_price") or 0
                p_3ago = data[i - 3].get(f"{prefix}_price") or 0
                feat["ret_15m"] = math.log(p_now / p_3ago) if p_now > 0 and p_3ago > 0 else 0.0
            else:
                feat["ret_15m"] = 0.0

            fg = hourly.get("fear_greed_value")
            feat["fear_greed"] = (fg - 50) / 50 if fg is not None else 0.0

            atr = hourly.get(f"{prefix}_atr_pct")
            feat["atr_pct"] = atr if atr is not None else 0.0

            flow = match_flow(ts, flow_data, prefix)
            br = flow.get("buy_ratio")
            feat["flow_buy_ratio"] = (br - 0.5) * 2 if br is not None else 0.0

            feat["flow_count_z"] = flow.get("count_zscore", 0) or 0.0
            feat["flow_size_z"] = flow.get("size_zscore", 0) or 0.0

            signal = flow.get("signal", "NORMAL")
            feat["flow_panic"] = 1.0 if signal == "PANIC" else 0.0
            feat["flow_fomo"] = 1.0 if signal == "FOMO" else 0.0

            bv_now = flow.get("buy_volume", 0) or 0
            sv_now = flow.get("sell_volume", 0) or 0
            cvd_delta = bv_now - sv_now
            total_vol = bv_now + sv_now
            feat["cvd_norm"] = cvd_delta / (total_vol + 1)

            if i >= 1:
                btc_ret_prev = data[i - 1].get("btc_ret_5m", 0) or 0
                eth_ret_prev = data[i - 1].get("eth_ret_5m", 0) or 0
                feat["btc_eth_diverg"] = btc_ret_prev - eth_ret_prev
            else:
                btc_ret_prev = 0.0
                eth_ret_prev = 0.0
                feat["btc_eth_diverg"] = 0.0

            feat["other_ret_lag1"] = eth_ret_prev if prefix == "btc" else btc_ret_prev

            if i >= 3:
                btc_p_now = data[i].get("btc_price") or 0
                btc_p_3ago = data[i - 3].get("btc_price") or 0
                eth_p_now = data[i].get("eth_price") or 0
                eth_p_3ago = data[i - 3].get("eth_price") or 0
                btc_ret15 = math.log(btc_p_now / btc_p_3ago) if btc_p_now > 0 and btc_p_3ago > 0 else 0.0
                eth_ret15 = math.log(eth_p_now / eth_p_3ago) if eth_p_now > 0 and eth_p_3ago > 0 else 0.0
                feat["btc_eth_diverg_15m"] = btc_ret15 - eth_ret15
            else:
                feat["btc_eth_diverg_15m"] = 0.0

            if i >= 6:
                p_now_30 = row.get(f"{prefix}_price") or 0
                p_6ago = data[i - 6].get(f"{prefix}_price") or 0
                feat["ret_30m"] = math.log(p_now_30 / p_6ago) if p_now_30 > 0 and p_6ago > 0 else 0.0
            else:
                feat["ret_30m"] = 0.0

            feat["acceleration"] = feat["ret_lag1"] - feat["ret_lag2"]

            ups = 0
            total_periods = 0
            for lag_i in range(max(0, i - 5), i + 1):
                r_val = data[lag_i].get(f"{prefix}_ret_5m")
                if r_val is not None:
                    total_periods += 1
                    if r_val > 0:
                        ups += 1
            feat["rsi_proxy"] = (ups / total_periods - 0.5) * 2 if total_periods > 0 else 0.0

            feat["vol_x_momentum"] = feat["log_volume"] * feat["ret_lag1"]

            feat["panic_x_volume"] = feat["flow_panic"] * feat["log_volume"]

            feat["fomo_x_volume"] = feat["flow_fomo"] * feat["log_volume"]

            feat["cvd_x_momentum"] = feat["cvd_norm"] * feat["ret_lag1"]

            recent_rets = []
            for lag_i in range(max(0, i - 6), i):
                r_val = data[lag_i].get(f"{prefix}_ret_5m")
                if r_val is not None:
                    recent_rets.append(r_val)
            if len(recent_rets) >= 3:
                mean_r = sum(recent_rets) / len(recent_rets)
                feat["realized_vol"] = math.sqrt(
                    sum((r - mean_r) ** 2 for r in recent_rets) / len(recent_rets)
                )
            else:
                feat["realized_vol"] = 0.0

            feat["vol_regime"] = feat["realized_vol"] * abs(feat["ret_lag1"])

            feat["egarch_vol"] = data[i].get(f"{prefix}_egarch_vol") or feat["realized_vol"]

            btc_evol = data[i].get("btc_egarch_vol") or 0
            eth_evol = data[i].get("eth_egarch_vol") or 0
            feat["btc_eth_vol_ratio"] = (btc_evol / eth_evol - 1) if eth_evol > 0 and btc_evol > 0 else 0.0

            if len(recent_rets) >= 3 and feat["realized_vol"] > 0:
                mean_r = sum(recent_rets) / len(recent_rets)
                skew_num = sum((r - mean_r) ** 3 for r in recent_rets) / len(recent_rets)
                feat["skewness"] = skew_num / (feat["realized_vol"] ** 3 + 1e-10)
            else:
                feat["skewness"] = 0.0

            feat["rolling_mean_ret"] = (
                sum(recent_rets) / len(recent_rets) if recent_rets else 0.0
            )

            feat["autocorr"] = feat["ret_lag1"] * feat["ret_lag2"]

            try:
                hour = int(ts[11:13])
                minute = int(ts[14:16])
                feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
                feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)

                from datetime import datetime as dt
                day_of_week = dt.strptime(ts[:10], "%Y-%m-%d").weekday()
                feat["dow_sin"] = math.sin(2 * math.pi * day_of_week / 7)
                feat["dow_cos"] = math.cos(2 * math.pi * day_of_week / 7)

                hours_since_settlement = hour % 8 + minute / 60
                feat["funding_proximity"] = math.cos(
                    2 * math.pi * hours_since_settlement / 8
                )

                feat["minute_sin"] = math.sin(2 * math.pi * minute / 60)
                feat["minute_cos"] = math.cos(2 * math.pi * minute / 60)

                min_to_round = min(minute, 60 - minute, abs(minute - 30))
                feat["round_hour_prox"] = math.exp(-min_to_round / 5)

                if 8 <= hour < 14:
                    feat["session_eu"] = 1.0
                    feat["session_overlap"] = 1.0 if 8 <= hour < 10 else 0.0
                elif 14 <= hour < 22:
                    feat["session_eu"] = 0.0
                    feat["session_overlap"] = 1.0 if 14 <= hour < 16 else 0.0
                else:
                    feat["session_eu"] = 0.0
                    feat["session_overlap"] = 0.0
            except (ValueError, IndexError):
                feat["hour_sin"] = 0.0
                feat["hour_cos"] = 0.0
                feat["dow_sin"] = 0.0
                feat["dow_cos"] = 0.0
                feat["funding_proximity"] = 0.0
                feat["minute_sin"] = 0.0
                feat["minute_cos"] = 0.0
                feat["round_hour_prox"] = 0.0
                feat["session_eu"] = 0.0
                feat["session_overlap"] = 0.0

            flow_win = get_flow_window(flow_by_sym, prefix, flow_idx[prefix], ts, 10)
            feat["vpin"] = compute_vpin(flow_win, n_buckets=6)

            feat["entropy"] = compute_entropy(flow_win, n=10)

            cur_ret = row.get(f"{prefix}_ret_5m", 0) or 0
            cur_vol = row.get(f"{prefix}_volume_5m", 0) or 0
            feat["kyle_lambda"] = compute_kyle_lambda(cur_ret, cur_vol)

            feat["gex"] = row.get(f"{prefix}_gex", 0) or 0.0

            feat["iv_rv_spread"] = row.get(f"{prefix}_options_skew", 0) or 0.0

            if i >= 1:
                ob_prev = data[i - 1].get(f"{prefix}_ob_imb")
                ob_now = row.get(f"{prefix}_ob_imb")
                if ob_prev is not None and ob_now is not None:
                    feat["ob_velocity"] = ob_now - ob_prev
                else:
                    feat["ob_velocity"] = 0.0
            else:
                feat["ob_velocity"] = 0.0

            cur_price = row.get(f"{prefix}_price", 0) or 0
            dollar_vol = cur_vol * cur_price if cur_price > 0 else 0
            feat["amihud"] = compute_amihud(cur_ret, dollar_vol)

            feat["granger_sol_eth"] = row.get("sol_eth_lag_ret", 0) or 0.0

            feat["corr_breakdown"] = compute_correlation_breakdown(data, i, window=12)

            feat["fractal_dim"] = compute_fractal_dimension(data, i, prefix, window=12)

            feat["gas_fee"] = row.get("gas_fee", 0) or 0.0

            bybit_col = f"{prefix}_bybit_price"
            bybit_p = row.get(bybit_col) or 0.0
            cur_p = row.get(f"{prefix}_price") or 0.0
            if bybit_p > 0 and cur_p > 0:
                feat["cross_ex_premium"] = (cur_p - bybit_p) / bybit_p * 100
            else:
                feat["cross_ex_premium"] = 0.0

            if prefix == "eth":
                sol_now = row.get("sol_price") or 0.0
                if i >= 1:
                    sol_prev = data[i - 1].get("sol_price") or 0.0
                    feat["granger_sol_eth"] = (
                        math.log(sol_now / sol_prev)
                        if sol_now > 0 and sol_prev > 0
                        else 0.0
                    )
                else:
                    feat["granger_sol_eth"] = 0.0
            else:
                feat["granger_sol_eth"] = row.get("granger_sol_eth", 0) or 0.0

            feat["max_pain"] = row.get(f"{prefix}_max_pain", 0) or 0.0

            for alt in ("bnb", "xrp", "link"):
                col = f"{alt}_price"
                now_p = row.get(col) or 0.0
                prev_p = data[i - 1].get(col) if i >= 1 else 0.0
                prev_p = prev_p or 0.0
                feat[f"{alt}_ret_lag1"] = (
                    math.log(now_p / prev_p) if now_p > 0 and prev_p > 0 else 0.0
                )

            btcd_now  = row.get("btc_dominance") or 0.0
            btcd_prev = (data[i - 1].get("btc_dominance") or 0.0) if i >= 1 else 0.0
            feat["btc_dominance"]        = btcd_now
            feat["btc_dominance_change"] = btcd_now - btcd_prev

            feat["social_sentiment"] = row.get("social_sentiment") or 0.0
            feat["social_volume"] = row.get("social_volume") or 0.0
            feat["trends_interest"] = (row.get("trends_interest") or 50.0) / 100.0

            if i >= 1:
                prev_sent = data[i - 1].get("social_sentiment") or 0.0
                feat["social_sentiment_change"] = feat["social_sentiment"] - prev_sent
            else:
                feat["social_sentiment_change"] = 0.0

            lookback = min(12, i)
            if lookback >= 2:
                avg_sent = sum((data[i - j].get("social_sentiment") or 0.0) for j in range(1, lookback + 1)) / lookback
                feat["sentiment_surprise"] = feat["social_sentiment"] - avg_sent
            else:
                feat["sentiment_surprise"] = 0.0

            feat["whale_tx_count"] = row.get("whale_tx_count") or 0.0
            feat["whale_tx_volume"] = row.get("whale_tx_volume") or 0.0

            feat["sp500_ret"] = row.get("sp500_ret_5m") or 0.0
            feat["nasdaq_ret"] = row.get("nasdaq_ret_5m") or 0.0

            tech = compute_technical_indicators(data, i, prefix)
            for tech_key, tech_val in tech.items():
                feat[tech_key] = tech_val

            r1 = feat["ret_lag1"]
            r2 = feat["ret_lag2"]
            feat["ret_lag1_sq"] = r1 * r1
            feat["ret_lag2_sq"] = r2 * r2
            feat["abs_ret_lag1"] = abs(r1)
            feat["abs_ret_lag2"] = abs(r2)
            feat["log_abs_ret_lag1"] = math.log(abs(r1) + 1e-6)
            cr = row.get(f"{prefix}_cascade_risk") or 0.0
            feat["cascade_risk_sq"] = cr * cr

            feat["ret1_x_cascade"] = r1 * cr
            feat["ret1_x_volume"] = r1 * feat["log_volume"]
            feat["ret1_x_obimb"] = r1 * feat["ob_imb"]
            feat["ret1_x_hurst"] = r1 * (feat["hurst"] - 0.5)
            feat["ret1_x_bb"] = r1 * feat["bb_position"]
            feat["ret1_x_super"] = r1 * feat["supertrend"]
            feat["macd_x_vol"] = feat["macd"] * feat["realized_vol"]
            feat["fg_x_cascade"] = feat.get("fear_greed", 0) * cr
            feat["sent_x_ret1"] = feat.get("social_sentiment", 0) * r1
            feat["mtf_x_ret1"] = feat["mtf_momentum_strength"] * r1

            feat["trade_arrival"] = flow.get("trades_per_min", 0) or 0.0

            feat["mempool_size"] = row.get("mempool_size") or 0.0
            feat["active_addresses"] = row.get("active_addresses") or 0.0
            feat["btc_tx_volume"] = row.get("btc_tx_volume") or 0.0

            realized_cap = row.get("realized_cap") or 0.0
            tx_vol = feat["btc_tx_volume"]
            feat["nvt_ratio"] = realized_cap / (tx_vol + 1) if tx_vol > 0 else 0.0

            feat["mempool_fee"] = row.get("mempool_fee") or 0.0

            feat["realized_cap"] = realized_cap

            feat["wmid_dev"] = row.get(f"{prefix}_wmid_dev") or 0.0
            feat["price_impact"] = math.log(max(row.get(f"{prefix}_price_impact") or 1, 1))
            feat["queue_imb"] = row.get(f"{prefix}_queue_imb") or 0.0

            feat["stable_inflow"] = 0.0

            feat["target"] = row.get(f"{prefix}_ret_5m", 0)
            feat["timestamp"] = ts
            feat["symbol"] = prefix

            enriched.append(feat)

    return enriched

def train_ols(features: list[dict], symbol: str) -> dict:
    sym_data = [f for f in features if f["symbol"] == symbol]

    if len(sym_data) < 30:
        log.warning(f"{symbol.upper()}: только {len(sym_data)} наблюдений, нужно минимум 30")
        return {"error": f"Недостаточно данных: {len(sym_data)} < 30"}

    feature_names = [
        "ob_imb", "log_volume", "oi_change_pct",
        "whale_imb", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5", "ret_lag6", "spread_pct",
        "ret_15m", "ret_30m",
        "fear_greed", "atr_pct",
        "flow_buy_ratio", "flow_count_z", "flow_size_z",
        "flow_panic", "flow_fomo",
        "cvd_norm", "btc_eth_diverg", "btc_eth_diverg_15m", "other_ret_lag1",
        "vpin", "entropy", "kyle_lambda", "gex", "iv_rv_spread",
        "ob_velocity", "amihud", "granger_sol_eth",
        "corr_breakdown", "fractal_dim", "gas_fee", "stable_inflow",
        "cross_ex_premium", "max_pain",
        "bnb_ret_lag1", "xrp_ret_lag1", "link_ret_lag1",
        "btc_dominance", "btc_dominance_change",
        "social_sentiment", "sentiment_surprise", "social_volume", "trends_interest", "social_sentiment_change",
        "whale_tx_count", "whale_tx_volume", "sp500_ret", "nasdaq_ret",
        "ema_diff", "macd", "macd_signal", "bb_position",
        "obv_change", "vwap_dev", "roll_measure", "roll_impact",
        "hurst", "macd_1h", "macd_15m",
        "ichimoku_conv", "ichimoku_base", "ichimoku_span_a",
        "supertrend", "keltner_pos",
        "volume_poc_dist", "price_efficiency",
        "volume_poc_1h", "rolling_sharpe_6", "rolling_sharpe_12",
        "volume_trend", "price_pos_30m", "high_low_ratio",
        "mtf_momentum_strength", "dir_persistence",
        "price_vs_ema5", "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
        "momentum_squeeze", "path_efficiency_30", "path_efficiency_60",
        "fisher_transform", "tsi", "vol_conf_momentum", "momentum_zscore",
        "ret_lag1_sq", "ret_lag2_sq", "abs_ret_lag1", "abs_ret_lag2",
        "log_abs_ret_lag1", "cascade_risk_sq",
        "ret1_x_cascade", "ret1_x_volume", "ret1_x_obimb", "ret1_x_hurst",
        "ret1_x_bb", "ret1_x_super", "macd_x_vol", "fg_x_cascade",
        "sent_x_ret1", "mtf_x_ret1",
        "trade_arrival", "mempool_size", "active_addresses", "btc_tx_volume",
        "nvt_ratio", "mempool_fee", "realized_cap",
        "wmid_dev", "price_impact", "queue_imb",
        "minute_sin", "minute_cos", "round_hour_prox",
        "session_eu", "session_overlap",
        "acceleration", "rsi_proxy",
        "vol_x_momentum", "panic_x_volume", "fomo_x_volume", "cvd_x_momentum",
        "realized_vol", "egarch_vol", "btc_eth_vol_ratio", "vol_regime",
        "skewness", "rolling_mean_ret", "autocorr",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "funding_proximity",
    ]

    X = np.array([[row[f] for f in feature_names] for row in sym_data], dtype=np.float64)
    y = np.array([row["target"] for row in sym_data], dtype=np.float64)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    if len(y) < 30:
        log.warning(f"{symbol.upper()}: после очистки осталось {len(y)} наблюдений")
        return {"error": f"Недостаточно чистых данных: {len(y)} < 30"}

    import pandas as pd
    X_df = pd.DataFrame(X, columns=feature_names)

    variances = X_df.var()
    valid_features = [f for f in feature_names if variances[f] > 1e-12]
    dropped = [f for f in feature_names if f not in valid_features]
    if dropped:
        log.warning(f"{symbol.upper()}: дропнуто {len(dropped)} фич с нулевой дисперсией: {dropped[:5]}...")

    X_df = X_df[valid_features]
    X_df = sm.add_constant(X_df)
    col_names = list(X_df.columns)

    model = sm.OLS(y, X_df).fit()

    coef = {name: float(model.params[name]) for name in model.params.index}
    pvalues = {name: float(model.pvalues[name]) for name in model.pvalues.index}

    feature_names = valid_features

    significant = [name for name in feature_names if pvalues.get(name, 1) < 0.1]

    result = {
        "full": {
            "coef": coef,
            "pvalues": pvalues,
            "adj_r2": float(model.rsquared_adj),
            "r2": float(model.rsquared),
            "n_obs": int(model.nobs),
            "features": feature_names,
        }
    }

    if significant and len(significant) < len(feature_names):
        idx = [0] + [feature_names.index(s) + 1 for s in significant]
        X_red = X[:, idx]
        red_names = ["const"] + significant

        model_red = sm.OLS(y, X_red).fit()
        result["reduced"] = {
            "coef": {name: float(model_red.params[i]) for i, name in enumerate(red_names)},
            "pvalues": {name: float(model_red.pvalues[i]) for i, name in enumerate(red_names)},
            "adj_r2": float(model_red.rsquared_adj),
            "r2": float(model_red.rsquared),
            "n_obs": int(model_red.nobs),
            "features": significant,
        }

    log.info(
        f"{symbol.upper()} OLS: n={model.nobs:.0f}, R²={model.rsquared:.4f}, "
        f"adj_R²={model.rsquared_adj:.4f}, "
        f"значимые: {significant or 'нет (p<0.1)'}"
    )

    return result

def train_rf(features: list[dict], symbol: str) -> dict:
    sym_data = [f for f in features if f["symbol"] == symbol]

    if len(sym_data) < 50:
        log.warning(f"{symbol.upper()} RF: только {len(sym_data)} наблюдений")
        return {"error": f"Недостаточно данных для RF: {len(sym_data)} < 50"}

    feature_names = [
        "ob_imb", "log_volume", "oi_change_pct",
        "whale_imb", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5", "ret_lag6", "spread_pct",
        "ret_15m", "ret_30m", "fear_greed", "atr_pct",
        "flow_buy_ratio", "flow_count_z", "flow_size_z",
        "flow_panic", "flow_fomo",
        "cvd_norm", "btc_eth_diverg", "btc_eth_diverg_15m", "other_ret_lag1",
        "vpin", "entropy", "kyle_lambda", "gex", "iv_rv_spread",
        "ob_velocity", "amihud", "granger_sol_eth",
        "corr_breakdown", "fractal_dim", "gas_fee", "stable_inflow",
        "cross_ex_premium", "max_pain",
        "bnb_ret_lag1", "xrp_ret_lag1", "link_ret_lag1",
        "btc_dominance", "btc_dominance_change",
        "social_sentiment", "sentiment_surprise", "social_volume", "trends_interest", "social_sentiment_change",
        "whale_tx_count", "whale_tx_volume", "sp500_ret", "nasdaq_ret",
        "ema_diff", "macd", "macd_signal", "bb_position",
        "obv_change", "vwap_dev", "roll_measure", "roll_impact",
        "hurst", "macd_1h", "macd_15m",
        "ichimoku_conv", "ichimoku_base", "ichimoku_span_a",
        "supertrend", "keltner_pos",
        "volume_poc_dist", "price_efficiency",
        "volume_poc_1h", "rolling_sharpe_6", "rolling_sharpe_12",
        "volume_trend", "price_pos_30m", "high_low_ratio",
        "mtf_momentum_strength", "dir_persistence",
        "price_vs_ema5", "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
        "momentum_squeeze", "path_efficiency_30", "path_efficiency_60",
        "fisher_transform", "tsi", "vol_conf_momentum", "momentum_zscore",
        "ret_lag1_sq", "ret_lag2_sq", "abs_ret_lag1", "abs_ret_lag2",
        "log_abs_ret_lag1", "cascade_risk_sq",
        "ret1_x_cascade", "ret1_x_volume", "ret1_x_obimb", "ret1_x_hurst",
        "ret1_x_bb", "ret1_x_super", "macd_x_vol", "fg_x_cascade",
        "sent_x_ret1", "mtf_x_ret1",
        "trade_arrival", "mempool_size", "active_addresses", "btc_tx_volume",
        "nvt_ratio", "mempool_fee", "realized_cap",
        "wmid_dev", "price_impact", "queue_imb",
        "minute_sin", "minute_cos", "round_hour_prox",
        "session_eu", "session_overlap",
        "acceleration", "rsi_proxy",
        "vol_x_momentum", "panic_x_volume", "fomo_x_volume", "cvd_x_momentum",
        "realized_vol", "egarch_vol", "btc_eth_vol_ratio", "vol_regime", "skewness", "rolling_mean_ret", "autocorr",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "funding_proximity",
    ]

    X = np.array([[row[f] for f in feature_names] for row in sym_data], dtype=np.float64)
    y = np.array([row["target"] for row in sym_data], dtype=np.float64)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    if len(y) < 50:
        return {"error": f"Недостаточно чистых данных для RF: {len(y)} < 50"}

    tscv = TimeSeriesSplit(n_splits=5)

    candidates = {
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features=0.4,
            random_state=42,
            n_jobs=-1,
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.01,
            min_samples_split=50,
            min_samples_leaf=25,
            subsample=0.7,
            random_state=42,
        ),
    }

    best_name = None
    best_score = -999
    best_model = None

    for name, model in candidates.items():
        cv = cross_val_score(model, X, y, cv=tscv, scoring="r2")
        mean_cv = float(np.mean(cv))
        log.info(f"{symbol.upper()} {name}: CV_R2={mean_cv:.4f} (+/-{np.std(cv):.4f})")
        if mean_cv > best_score:
            best_score = mean_cv
            best_name = name
            best_model = model

    log.info(f"{symbol.upper()}: best ML model = {best_name} (CV_R2={best_score:.4f})")

    best_model.fit(X, y)
    rf = best_model

    y_pred = rf.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)

    cv_scores = cross_val_score(best_model, X, y, cv=tscv, scoring="r2")
    cv_mae = cross_val_score(best_model, X, y, cv=tscv, scoring="neg_mean_absolute_error")

    importances = {name: float(imp) for name, imp in
                   zip(feature_names, rf.feature_importances_)}
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

    RF_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RF_DIR / f"rf_{symbol}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf, "features": feature_names}, f)

    result = {
        "model_type": best_name,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "cv_r2_scores": [float(s) for s in cv_scores],
        "cv_mae_mean": float(-np.mean(cv_mae)),
        "n_obs": len(y),
        "features": feature_names,
        "feature_importance": importances,
        "top_features": top_features,
        "model_path": str(model_path),
    }

    log.info(
        f"{symbol.upper()} BEST ({best_name}): n={len(y)}, "
        f"train_R²={train_r2:.4f}, "
        f"CV_R²={np.mean(cv_scores):.4f} (+/-{np.std(cv_scores):.4f}), "
        f"top: {', '.join(f[0] for f in top_features[:3])}"
    )

    return result

def print_report(ols_results: dict, rf_results: dict):
    print("\n" + "=" * 70)
    print("  REPORT: 5-MIN INTRADAY MODELS")
    print("=" * 70)

    for symbol in ("btc", "eth"):
        sym_res = ols_results.get(symbol, {})
        if "error" in sym_res:
            print(f"\n  {symbol.upper()} OLS: {sym_res['error']}")
        else:
            full = sym_res.get("full", {})
            reduced = sym_res.get("reduced")

            print(f"\n{'-' * 35}")
            print(f"  {symbol.upper()} -- OLS Full model")
            print(f"{'-' * 35}")
            print(f"  Observations: {full.get('n_obs', 0)}")
            print(f"  R2:           {full.get('r2', 0):.6f}")
            print(f"  Adj R2:       {full.get('adj_r2', 0):.6f}")
            print()

            coef = full.get("coef", {})
            pval = full.get("pvalues", {})
            print(f"  {'Feature':<20} {'Coef':>12} {'p-value':>10}  Sig?")
            print(f"  {'-' * 55}")
            for name in ["const"] + full.get("features", []):
                c = coef.get(name, 0)
                p = pval.get(name, 1)
                sig = "*" if p < 0.05 else ("~" if p < 0.1 else "")
                print(f"  {name:<20} {c:>12.6f} {p:>10.4f}  {sig}")

            if reduced:
                print(f"\n  {symbol.upper()} -- OLS Reduced")
                print(f"  Adj R2: {reduced.get('adj_r2', 0):.6f}")
                print(f"  Features: {', '.join(reduced.get('features', []))}")

        rf_res = rf_results.get(symbol, {})
        if "error" in rf_res:
            print(f"\n  {symbol.upper()} RF: {rf_res['error']}")
        else:
            print(f"\n{'-' * 35}")
            print(f"  {symbol.upper()} -- Random Forest")
            print(f"{'-' * 35}")
            print(f"  Observations:  {rf_res.get('n_obs', 0)}")
            print(f"  Train R2:      {rf_res.get('train_r2', 0):.6f}")
            print(f"  CV R2 (mean):  {rf_res.get('cv_r2_mean', 0):.6f} "
                  f"(+/- {rf_res.get('cv_r2_std', 0):.4f})")
            print(f"  CV R2 (folds): {rf_res.get('cv_r2_scores', [])}")
            print(f"  CV MAE:        {rf_res.get('cv_mae_mean', 0):.6f}")
            print()
            print(f"  Top features:")
            for fname, imp in rf_res.get("top_features", []):
                bar = "#" * int(imp * 100)
                print(f"    {fname:<20} {imp:.4f}  {bar}")

        ols_r2 = sym_res.get("full", {}).get("adj_r2", 0) if "error" not in sym_res else 0
        rf_cv = rf_res.get("cv_r2_mean", 0) if "error" not in rf_res else 0
        print(f"\n  >> {symbol.upper()} COMPARISON:")
        print(f"     OLS adj_R2:  {ols_r2:.4f}")
        print(f"     RF CV_R2:    {rf_cv:.4f}")
        if rf_cv > ols_r2:
            improvement = ((rf_cv - ols_r2) / max(abs(ols_r2), 0.001)) * 100
            print(f"     RF better by {improvement:.0f}%")
        else:
            print(f"     OLS better or equal")

    print(f"\n{'=' * 70}")
    print(f"  OLS coefficients: {OUT_FILE}")
    print(f"  RF models:        {RF_DIR}/")
    print(f"{'=' * 70}\n")

def main():
    if not DB_PATH.exists():
        log.error(f"БД не найдена: {DB_PATH}")
        return

    data = load_data(str(DB_PATH))
    if len(data) < 30:
        log.error(f"Слишком мало данных ({len(data)}). Нужно минимум 30 записей.")
        return

    hourly_data = load_hourly_data(str(HOURLY_PATH))
    flow_data = load_flow_data(str(FLOW_PATH))

    features = compute_features(data, hourly_data, flow_data)

    ols_results = {}
    for symbol in ("btc", "eth"):
        ols_results[symbol] = train_ols(features, symbol)

    rf_results = {}
    for symbol in ("btc", "eth"):
        rf_results[symbol] = train_rf(features, symbol)

    results = ols_results
    output = {}
    for symbol in ("btc", "eth"):
        sym_res = results[symbol]
        if "error" in sym_res:
            continue

        models = {}
        has_reduced = "reduced" in sym_res

        models["full"] = {
            "adj_r2": sym_res["full"]["adj_r2"],
            "coef": sym_res["full"]["coef"],
            "n_obs": sym_res["full"]["n_obs"],
            "features": sym_res["full"]["features"],
            "pvalues": sym_res["full"]["pvalues"],
        }

        if has_reduced:
            models["reduced"] = {
                "adj_r2": sym_res["reduced"]["adj_r2"],
                "coef": sym_res["reduced"]["coef"],
                "n_obs": sym_res["reduced"]["n_obs"],
                "features": sym_res["reduced"]["features"],
                "pvalues": sym_res["reduced"]["pvalues"],
            }

        rf_res = rf_results.get(symbol, {})
        if "error" not in rf_res:
            models["rf"] = {
                "train_r2": rf_res["train_r2"],
                "cv_r2_mean": rf_res["cv_r2_mean"],
                "cv_r2_std": rf_res["cv_r2_std"],
                "cv_mae_mean": rf_res["cv_mae_mean"],
                "n_obs": rf_res["n_obs"],
                "features": rf_res["features"],
                "feature_importance": rf_res["feature_importance"],
                "model_path": rf_res["model_path"],
            }

        output[symbol] = {
            "default_model": "ensemble",
            "models": models,
            "trained_at": datetime.now().isoformat(),
            "horizon": "5min",
        }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"Коэффициенты сохранены: {OUT_FILE}")

    print_report(ols_results, rf_results)

if __name__ == "__main__":
    main()
