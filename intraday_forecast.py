import json
import math
import logging
import pickle
import sqlite3
from pathlib import Path

import numpy as np

COEF_FILE    = Path("data/intraday_coefficients.json")
HOURLY_PATH  = Path("data/hourly.db")
FLOW_PATH    = Path("data/flow.db")
RF_DIR       = Path("data/rf_models")
INTRADAY_PATH = Path("data/intraday.db")

def compute_regime_adjustment(symbol: str, predicted_pct: float,
                               fear_greed: float, sentiment: float,
                               realized_vol: float, logger=None) -> dict:
    prefix = symbol.lower()
    adj = 0.0
    confidence_mult = 1.0
    regime_parts = []

    fg = fear_greed * 50 + 50 if abs(fear_greed) <= 1 else fear_greed
    is_fear  = fg < 30
    is_greed = fg > 70

    is_high_vol = realized_vol > 0.002

    is_bull = fg > 50

    sign = 1 if predicted_pct > 0 else -1

    if prefix == "btc":
        if is_bull:
            adj += 0.003 * sign
            regime_parts.append("BULL")

        if is_bull and sentiment < -0.2:
            if predicted_pct < 0:
                confidence_mult *= 0.7
                regime_parts.append("neg_nullified")

        if is_high_vol and abs(sentiment) > 0.3:
            adj += sentiment * 0.005 * sign
            confidence_mult *= 1.2
            regime_parts.append("HV+sent")

        if is_fear and sentiment > 0.2 and predicted_pct > 0:
            confidence_mult *= 1.3
            regime_parts.append("FEAR+pos_sent")

        if is_greed:
            confidence_mult *= 0.85
            regime_parts.append("GREED_caution")

    elif prefix == "eth":
        if abs(sentiment) > 0.2:
            adj += sentiment * 0.003
            regime_parts.append(f"sent={'pos' if sentiment > 0 else 'neg'}")

        if is_greed and sentiment > 0.2:
            confidence_mult *= 1.2
            regime_parts.append("GREED+sent_boost")

        if is_fear:
            adj += 0.002
            confidence_mult *= 1.1
            regime_parts.append("FEAR_pos")

        if is_greed:
            confidence_mult *= 0.9
            regime_parts.append("GREED_neg")

        if is_high_vol:
            confidence_mult *= 0.9
            regime_parts.append("HV_caution")

    adjusted_pct = predicted_pct * confidence_mult + adj
    regime_str = "+".join(regime_parts) if regime_parts else "neutral"

    if logger:
        logger.debug(
            f"Regime [{prefix}]: FG={fg:.0f}, sent={sentiment:.2f}, "
            f"vol={'HI' if is_high_vol else 'LO'} → {regime_str}, "
            f"conf={confidence_mult:.2f}, adj={adj:+.4f}%"
        )

    return {
        "adjusted_pct": adjusted_pct,
        "regime": regime_str,
        "confidence_mult": confidence_mult,
        "fear_greed_raw": fg,
        "is_bull": is_bull,
        "is_high_vol": is_high_vol,
    }

_COEF_CACHE: dict = {}
_RF_CACHE: dict = {}

def _compute_tech_indicators(rows_asc: list[dict], prefix: str) -> dict:
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
        "mtf_momentum_strength": 0.0, "dir_persistence": 0.0,
        "price_vs_ema5": 0.0, "price_vs_ema20": 0.0,
        "price_vs_ema50": 0.0, "price_vs_ema200": 0.0,
        "momentum_squeeze": 1.0, "path_efficiency_30": 0.0,
        "path_efficiency_60": 0.0, "fisher_transform": 0.0,
        "tsi": 0.0, "vol_conf_momentum": 0.0, "momentum_zscore": 0.0,
    }

    prices = []
    volumes = []
    for r in rows_asc:
        p = r.get(f"{prefix}_price")
        v = r.get(f"{prefix}_volume_5m")
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
        result["macd"] = (ema12 - ema26) / cur_price * 100
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
            cov = sum((rets[k] - mean_r) * (rets[k - 1] - mean_r) for k in range(1, n)) / (n - 1)
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
            if S > 0 and R > 0 and n > 2:
                result["hurst"] = math.log(R / S) / math.log(n / 2)
                result["hurst"] = max(0.1, min(0.9, result["hurst"]))
        except (ValueError, ZeroDivisionError):
            pass

    if len(prices) >= 40:
        prices_15m = prices[::3]
        if len(prices_15m) >= 26:
            result["macd_15m"] = (ema(prices_15m, 12) - ema(prices_15m, 26)) / cur_price * 100

    if len(prices) >= 60:
        prices_1h = prices[::12]
        if len(prices_1h) >= 26:
            result["macd_1h"] = (ema(prices_1h, 12) - ema(prices_1h, 26)) / cur_price * 100
        elif len(prices_1h) >= 5:
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
            result["keltner_pos"] = max(-2, min(2, (cur_price - kc_mid) / (2 * atr)))

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
        atr_vals2 = [abs(prices[k] - prices[k - 1]) for k in range(len(prices) - 14, len(prices))]
        atr2 = sum(atr_vals2) / 14 if atr_vals2 else 0
    else:
        atr2 = 0

    if atr2 > 0:
        result["price_vs_ema5"] = (cur_price - ema(prices, 5)) / atr2
        result["price_vs_ema20"] = (cur_price - ema(prices, 20)) / atr2
        if len(prices) >= 50:
            result["price_vs_ema50"] = (cur_price - ema(prices, 50)) / atr2
        if len(prices) >= 60:
            result["price_vs_ema200"] = (cur_price - ema(prices, min(200, len(prices)))) / atr2

    if len(prices) >= 20 and atr2 > 0:
        sma20 = sum(prices[-20:]) / 20
        std20 = math.sqrt(sum((p - sma20) ** 2 for p in prices[-20:]) / 20)
        bb_width = 4 * std20
        kc_width = 4 * atr2
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
        rets_f = [math.log(prices[k] / prices[k - 1]) for k in range(len(prices) - 6, len(prices))]
        mean_f = sum(rets_f) / len(rets_f)
        std_f = math.sqrt(sum((r - mean_f) ** 2 for r in rets_f) / len(rets_f)) or 1e-10
        x = rets_f[-1] / (3 * std_f)
        x = max(-0.999, min(0.999, x))
        try:
            result["fisher_transform"] = 0.5 * math.log((1 + x) / (1 - x))
        except (ValueError, ZeroDivisionError):
            pass

    if len(prices) >= 26:
        pc = [prices[k] - prices[k - 1] for k in range(1, len(prices))]
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

def _load_coefficients() -> dict:
    global _COEF_CACHE
    if _COEF_CACHE:
        return _COEF_CACHE
    if not COEF_FILE.exists():
        raise FileNotFoundError(f"Файл коэффициентов не найден: {COEF_FILE}")
    with open(COEF_FILE, "r", encoding="utf-8") as f:
        _COEF_CACHE = json.load(f)
    return _COEF_CACHE

def _load_rf_model(symbol: str):
    global _RF_CACHE
    sym = symbol.lower()
    if sym in _RF_CACHE:
        return _RF_CACHE[sym]
    model_path = RF_DIR / f"rf_{sym}.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    _RF_CACHE[sym] = data
    return data

def reload_coefficients():
    global _COEF_CACHE, _RF_CACHE
    _COEF_CACHE = {}
    _RF_CACHE = {}
    return _load_coefficients()

def _get_latest_hourly(symbol: str) -> dict:
    if not HOURLY_PATH.exists():
        return {}
    try:
        prefix = symbol.lower()
        conn = sqlite3.connect(str(HOURLY_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT fear_greed_value, {atr} as atr_pct
            FROM hourly_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """.format(atr=f"{prefix}_atr_pct")).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}

def _get_latest_flow(symbol: str) -> dict:
    if not FLOW_PATH.exists():
        return {}
    try:
        sym_map = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
        target_sym = sym_map.get(symbol.lower(), "")
        conn = sqlite3.connect(str(FLOW_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT buy_ratio, count_zscore, size_zscore, signal,
                   buy_volume, sell_volume
            FROM flow_snapshots
            WHERE exchange = 'all' AND symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (target_sym,)).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}

def get_latest_features(db_path: str, symbol: str) -> dict:
    prefix = symbol.lower()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM intraday_snapshots
        WHERE {price} IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 60
    """.format(price=f"{prefix}_price")).fetchall()

    sent_avg_row = conn.execute("""
        SELECT AVG(social_sentiment) FROM (
            SELECT social_sentiment FROM intraday_snapshots
            WHERE social_sentiment IS NOT NULL
            ORDER BY timestamp DESC LIMIT 12
        )
    """).fetchone()
    _sentiment_avg = float(sent_avg_row[0]) if sent_avg_row and sent_avg_row[0] else 0.0

    conn.close()

    if not rows:
        return {"error": "Нет данных в intraday.db"}

    rows = [dict(r) for r in rows]
    current = rows[0]

    rows_asc = list(reversed(rows))
    tech_indicators = _compute_tech_indicators(rows_asc, prefix)

    ob_imb_raw = current.get(f"{prefix}_ob_imb")
    ob_imb = (ob_imb_raw - 0.5) * 2 if ob_imb_raw is not None else 0.0

    vol = current.get(f"{prefix}_volume_5m")
    log_volume = math.log(vol) if vol and vol > 0 else 0.0

    oi_now = current.get(f"{prefix}_oi")
    if len(rows) >= 2:
        oi_prev = rows[1].get(f"{prefix}_oi")
        if oi_now and oi_prev and oi_prev > 0:
            oi_change_pct = (oi_now - oi_prev) / oi_prev * 100
        else:
            oi_change_pct = 0.0
    else:
        oi_change_pct = 0.0

    wb = current.get(f"{prefix}_whale_buy", 0) or 0
    ws = current.get(f"{prefix}_whale_sell", 0) or 0
    whale_imb = (wb - ws) / (wb + ws + 1)

    ret_lag1 = rows[1].get(f"{prefix}_ret_5m", 0) or 0 if len(rows) >= 2 else 0.0
    ret_lag2 = rows[2].get(f"{prefix}_ret_5m", 0) or 0 if len(rows) >= 3 else 0.0
    ret_lag3 = rows[3].get(f"{prefix}_ret_5m", 0) or 0 if len(rows) >= 4 else 0.0
    ret_lag4 = rows[4].get(f"{prefix}_ret_5m", 0) or 0 if len(rows) >= 5 else 0.0
    ret_lag5 = rows[5].get(f"{prefix}_ret_5m", 0) or 0 if len(rows) >= 6 else 0.0
    ret_lag6 = rows[6].get(f"{prefix}_ret_5m", 0) or 0 if len(rows) >= 7 else 0.0

    spread = current.get(f"{prefix}_spread_pct")
    spread_pct = spread if spread is not None else 0.0

    if len(rows) >= 4:
        p_cur_15 = rows[0].get(f"{prefix}_price") or 0
        p_3ago = rows[3].get(f"{prefix}_price") or 0
        ret_15m = math.log(p_cur_15 / p_3ago) if p_cur_15 > 0 and p_3ago > 0 else 0.0
    else:
        ret_15m = 0.0

    hourly = _get_latest_hourly(symbol)
    fg = hourly.get("fear_greed_value")
    fear_greed = (fg - 50) / 50 if fg is not None else 0.0

    atr = hourly.get("atr_pct")
    atr_pct = atr if atr is not None else 0.0

    flow = _get_latest_flow(symbol)
    br = flow.get("buy_ratio")
    flow_buy_ratio = (br - 0.5) * 2 if br is not None else 0.0

    flow_count_z = flow.get("count_zscore", 0) or 0.0
    flow_size_z = flow.get("size_zscore", 0) or 0.0

    signal = flow.get("signal", "NORMAL")
    flow_panic = 1.0 if signal == "PANIC" else 0.0
    flow_fomo = 1.0 if signal == "FOMO" else 0.0

    bv = flow.get("buy_volume", 0) or 0
    sv = flow.get("sell_volume", 0) or 0
    cvd_delta = bv - sv
    total_vol = bv + sv
    cvd_norm = cvd_delta / (total_vol + 1)

    if len(rows) >= 2:
        btc_ret_prev = rows[1].get("btc_ret_5m", 0) or 0
        eth_ret_prev = rows[1].get("eth_ret_5m", 0) or 0
        btc_eth_diverg = btc_ret_prev - eth_ret_prev
    else:
        btc_ret_prev = 0.0
        eth_ret_prev = 0.0
        btc_eth_diverg = 0.0

    other_ret_lag1 = eth_ret_prev if prefix == "btc" else btc_ret_prev

    try:
        hour = int(current["timestamp"][11:13])
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
    except (ValueError, IndexError, TypeError):
        hour_sin = 0.0
        hour_cos = 0.0

    if len(rows) >= 7:
        p_cur_30 = rows[0].get(f"{prefix}_price") or 0
        p_6ago = rows[6].get(f"{prefix}_price") or 0
        ret_30m = math.log(p_cur_30 / p_6ago) if p_cur_30 > 0 and p_6ago > 0 else 0.0
    else:
        ret_30m = 0.0

    if len(rows) >= 4:
        btc_p_now = rows[0].get("btc_price") or 0
        btc_p_3ago = rows[3].get("btc_price") or 0
        eth_p_now = rows[0].get("eth_price") or 0
        eth_p_3ago = rows[3].get("eth_price") or 0
        btc_r15 = math.log(btc_p_now / btc_p_3ago) if btc_p_now > 0 and btc_p_3ago > 0 else 0.0
        eth_r15 = math.log(eth_p_now / eth_p_3ago) if eth_p_now > 0 and eth_p_3ago > 0 else 0.0
        btc_eth_diverg_15m = btc_r15 - eth_r15
    else:
        btc_eth_diverg_15m = 0.0

    acceleration = ret_lag1 - ret_lag2
    recent_rets = []
    for r in rows[1:min(7, len(rows))]:
        rv = r.get(f"{prefix}_ret_5m")
        if rv is not None:
            recent_rets.append(rv)
    ups = sum(1 for r in recent_rets if r > 0)
    rsi_proxy = (ups / len(recent_rets) - 0.5) * 2 if recent_rets else 0.0
    rolling_mean_ret = sum(recent_rets) / len(recent_rets) if recent_rets else 0.0
    autocorr = ret_lag1 * ret_lag2

    if len(recent_rets) >= 3:
        mean_r = sum(recent_rets) / len(recent_rets)
        realized_vol = math.sqrt(sum((r - mean_r) ** 2 for r in recent_rets) / len(recent_rets))
        skew_num = sum((r - mean_r) ** 3 for r in recent_rets) / len(recent_rets)
        skewness = skew_num / (realized_vol ** 3 + 1e-10)
    else:
        realized_vol = 0.0
        skewness = 0.0
    vol_regime = realized_vol * abs(ret_lag1)

    egarch_vol = current.get(f"{prefix}_egarch_vol") or realized_vol
    vol_anomaly = current.get(f"{prefix}_vol_anomaly") or 0

    btc_evol = current.get("btc_egarch_vol") or 0
    eth_evol = current.get("eth_egarch_vol") or 0
    btc_eth_vol_ratio = (btc_evol / eth_evol - 1) if eth_evol > 0 and btc_evol > 0 else 0.0

    vol_x_momentum = log_volume * ret_lag1
    panic_x_volume = flow_panic * log_volume
    fomo_x_volume = flow_fomo * log_volume
    cvd_x_momentum = cvd_norm * ret_lag1

    try:
        minute = int(current["timestamp"][14:16])
        from datetime import datetime as dt
        day_of_week = dt.strptime(current["timestamp"][:10], "%Y-%m-%d").weekday()
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)
        hour = int(current["timestamp"][11:13])
        hours_since = hour % 8 + minute / 60
        funding_proximity = math.cos(2 * math.pi * hours_since / 8)
    except (ValueError, IndexError, TypeError):
        dow_sin = 0.0
        dow_cos = 0.0
        funding_proximity = 0.0

    gex = current.get(f"{prefix}_gex", 0) or 0.0
    iv_rv_spread = current.get(f"{prefix}_options_skew", 0) or 0.0
    max_pain = current.get(f"{prefix}_max_pain", 0) or 0.0
    gas_fee = current.get("gas_fee", 0) or 0.0

    ob_velocity = 0.0
    if len(rows) >= 2:
        ob_prev = rows[1].get(f"{prefix}_ob_imb")
        ob_now = current.get(f"{prefix}_ob_imb")
        if ob_prev is not None and ob_now is not None:
            ob_velocity = ob_now - ob_prev

    bybit_p = current.get(f"{prefix}_bybit_price") or 0.0
    cur_p = current.get(f"{prefix}_price") or 0.0
    cross_ex_premium = (cur_p - bybit_p) / bybit_p * 100 if bybit_p > 0 and cur_p > 0 else 0.0

    social_sentiment = current.get("social_sentiment") or 0.0
    social_volume = current.get("social_volume") or 0.0
    trends_interest = (current.get("trends_interest") or 50.0) / 100.0
    sentiment_surprise = social_sentiment - _sentiment_avg

    sp500_ret = current.get("sp500_ret_5m") or 0.0
    nasdaq_ret = current.get("nasdaq_ret_5m") or 0.0

    whale_tx_count = current.get("whale_tx_count") or 0.0
    whale_tx_volume = current.get("whale_tx_volume") or 0.0

    cascade_risk = current.get(f"{prefix}_cascade_risk") or 0.0

    ret_lag1_sq = ret_lag1 * ret_lag1
    ret_lag2_sq = ret_lag2 * ret_lag2
    abs_ret_lag1 = abs(ret_lag1)
    abs_ret_lag2 = abs(ret_lag2)
    log_abs_ret_lag1 = math.log(abs_ret_lag1 + 1e-6)
    cascade_risk_sq = cascade_risk * cascade_risk

    ret1_x_cascade = ret_lag1 * cascade_risk
    ret1_x_volume = ret_lag1 * log_volume
    ret1_x_obimb = ret_lag1 * ob_imb
    ret1_x_hurst = ret_lag1 * (tech_indicators.get("hurst", 0.5) - 0.5)
    ret1_x_bb = ret_lag1 * tech_indicators.get("bb_position", 0)
    ret1_x_super = ret_lag1 * tech_indicators.get("supertrend", 0)
    macd_x_vol = tech_indicators.get("macd", 0) * realized_vol
    fg_x_cascade = fear_greed * cascade_risk
    sent_x_ret1 = social_sentiment * ret_lag1
    mtf_x_ret1 = tech_indicators.get("mtf_momentum_strength", 0) * ret_lag1

    return {
        "ob_imb": ob_imb,
        "log_volume": log_volume,
        "oi_change_pct": oi_change_pct,
        "whale_imb": whale_imb,
        "ret_lag1": ret_lag1,
        "ret_lag2": ret_lag2,
        "ret_lag3": ret_lag3,
        "ret_lag4": ret_lag4,
        "ret_lag5": ret_lag5,
        "ret_lag6": ret_lag6,
        "spread_pct": spread_pct,
        "ret_15m": ret_15m,
        "ret_30m": ret_30m,
        "fear_greed": fear_greed,
        "atr_pct": atr_pct,
        "flow_buy_ratio": flow_buy_ratio,
        "flow_count_z": flow_count_z,
        "flow_size_z": flow_size_z,
        "flow_panic": flow_panic,
        "flow_fomo": flow_fomo,
        "cvd_norm": cvd_norm,
        "btc_eth_diverg": btc_eth_diverg,
        "btc_eth_diverg_15m": btc_eth_diverg_15m,
        "vpin": 0.0,
        "entropy": 0.0,
        "kyle_lambda": abs(ret_lag1) / math.log(max(current.get(f"{prefix}_volume_5m") or 1, 2)),
        "gex": gex,
        "iv_rv_spread": iv_rv_spread,
        "ob_velocity": ob_velocity,
        "amihud": abs(ret_lag1) * 1e6 / max((current.get(f"{prefix}_volume_5m") or 1) * cur_p, 1),
        "granger_sol_eth": 0.0,
        "corr_breakdown": 0.0,
        "fractal_dim": 1.5,
        "gas_fee": gas_fee,
        "stable_inflow": 0.0,
        "cross_ex_premium": cross_ex_premium,
        "max_pain": max_pain,
        "bnb_ret_lag1": 0.0,
        "xrp_ret_lag1": 0.0,
        "link_ret_lag1": 0.0,
        "btc_dominance": current.get("btc_dominance") or 0.0,
        "btc_dominance_change": 0.0,
        "social_sentiment": social_sentiment,
        "sentiment_surprise": sentiment_surprise,
        "social_volume": social_volume,
        "trends_interest": trends_interest,
        "social_sentiment_change": 0.0,
        "whale_tx_count": whale_tx_count,
        "whale_tx_volume": whale_tx_volume,
        "sp500_ret": sp500_ret,
        "nasdaq_ret": nasdaq_ret,
        "acceleration": acceleration,
        "rsi_proxy": rsi_proxy,
        "vol_x_momentum": vol_x_momentum,
        "panic_x_volume": panic_x_volume,
        "fomo_x_volume": fomo_x_volume,
        "cvd_x_momentum": cvd_x_momentum,
        "realized_vol": realized_vol,
        "egarch_vol": egarch_vol,
        "vol_anomaly": vol_anomaly,
        "btc_eth_vol_ratio": btc_eth_vol_ratio,
        "other_ret_lag1": other_ret_lag1,
        "cascade_risk": cascade_risk,
        "vol_regime": vol_regime,
        "skewness": skewness,
        "rolling_mean_ret": rolling_mean_ret,
        "autocorr": autocorr,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "funding_proximity": funding_proximity,
        **tech_indicators,
        "ret_lag1_sq": ret_lag1_sq,
        "ret_lag2_sq": ret_lag2_sq,
        "abs_ret_lag1": abs_ret_lag1,
        "abs_ret_lag2": abs_ret_lag2,
        "log_abs_ret_lag1": log_abs_ret_lag1,
        "cascade_risk_sq": cascade_risk_sq,
        "ret1_x_cascade": ret1_x_cascade,
        "ret1_x_volume": ret1_x_volume,
        "ret1_x_obimb": ret1_x_obimb,
        "ret1_x_hurst": ret1_x_hurst,
        "ret1_x_bb": ret1_x_bb,
        "ret1_x_super": ret1_x_super,
        "macd_x_vol": macd_x_vol,
        "fg_x_cascade": fg_x_cascade,
        "sent_x_ret1": sent_x_ret1,
        "mtf_x_ret1": mtf_x_ret1,
        "timestamp": current["timestamp"],
        "price": current.get(f"{prefix}_price"),
    }

def get_intraday_forecast(
    symbol: str,
    db_path: str = "data/intraday.db",
    model_name: str = None,
    logger: logging.Logger = None,
) -> dict:
    try:
        coefs_all = _load_coefficients()
    except FileNotFoundError as e:
        return {"error": str(e)}

    sym = symbol.lower()
    sym_data = coefs_all.get(sym, {})
    if not sym_data:
        return {"error": f"Нет коэффициентов для {symbol}"}

    if model_name is None:
        model_name = sym_data.get("default_model", "full")

    models = sym_data.get("models", {})

    if model_name not in models and model_name != "ensemble":
        return {"error": f"Модель {model_name} не найдена для {symbol}"}

    ols_key = "reduced" if "reduced" in models else "full"
    model = models.get(ols_key, models.get("full", {}))
    coef = model.get("coef", {})
    adj_r2 = model.get("adj_r2", 0)
    n_obs = model.get("n_obs", 0)

    features = get_latest_features(db_path, symbol)
    if "error" in features:
        return features

    cur_price = features.pop("price", None)
    ts = features.pop("timestamp", None)

    inputs = {"const": 1.0, **features}

    ols_model_name = "reduced" if "reduced" in models else "full"
    ols_coef = models.get(ols_model_name, models.get("full", {})).get("coef", coef)
    ols_r2 = models.get(ols_model_name, {}).get("adj_r2", adj_r2)

    ols_log_return = sum(
        ols_coef.get(var, 0) * inputs.get(var, 0.0)
        for var in ols_coef
    )
    ols_pct = (math.exp(ols_log_return) - 1) * 100

    rf_pct = None
    rf_data = _load_rf_model(symbol)
    if rf_data is not None:
        try:
            rf_model = rf_data["model"]
            rf_features = rf_data["features"]
            X_rf = np.array([[inputs.get(f, 0.0) for f in rf_features]], dtype=np.float64)
            rf_log_return = rf_model.predict(X_rf)[0]
            rf_pct = (math.exp(rf_log_return) - 1) * 100
        except Exception:
            rf_pct = None

    if model_name == "ensemble" and rf_pct is not None:
        rf_info = models.get("rf", {})
        rf_cv_r2 = rf_info.get("cv_r2_mean", 0)
        ols_adj_r2 = ols_r2

        total = max(abs(ols_adj_r2), 0.001) + max(abs(rf_cv_r2), 0.001)
        w_ols = max(abs(ols_adj_r2) / total, 0.2)
        w_rf = max(abs(rf_cv_r2) / total, 0.2)
        w_sum = w_ols + w_rf
        w_ols /= w_sum
        w_rf /= w_sum

        predicted_pct = w_ols * ols_pct + w_rf * rf_pct
        used_model = f"ensemble (OLS {w_ols:.0%} + RF {w_rf:.0%})"
        used_r2 = w_ols * ols_adj_r2 + w_rf * rf_cv_r2
    elif model_name == "rf" and rf_pct is not None:
        predicted_pct = rf_pct
        used_model = "rf"
        used_r2 = models.get("rf", {}).get("cv_r2_mean", 0)
    else:
        predicted_pct = ols_pct
        used_model = ols_model_name
        used_r2 = ols_r2

    regime_info = compute_regime_adjustment(
        symbol=symbol,
        predicted_pct=predicted_pct,
        fear_greed=inputs.get("fear_greed", 0),
        sentiment=inputs.get("social_sentiment", 0),
        realized_vol=inputs.get("realized_vol", 0),
        logger=logger,
    )
    raw_pct = predicted_pct
    predicted_pct = regime_info["adjusted_pct"]

    if logger:
        parts = [f"OLS={ols_pct:+.4f}%"]
        if rf_pct is not None:
            parts.append(f"RF={rf_pct:+.4f}%")
        parts.append(f"raw={raw_pct:+.4f}%")
        parts.append(f"regime={regime_info['regime']}")
        parts.append(f"final={predicted_pct:+.4f}%")
        logger.debug(
            f"Intraday [{symbol}/{used_model}]: {', '.join(parts)}"
        )

    return {
        "predicted_return": math.log(1 + predicted_pct / 100),
        "predicted_pct": predicted_pct,
        "predicted_price": cur_price * (1 + predicted_pct / 100) if cur_price else None,
        "current_price": cur_price,
        "model": used_model,
        "adj_r2": used_r2,
        "n_obs": n_obs,
        "symbol": symbol.upper(),
        "timestamp": ts,
        "inputs": inputs,
        "ols_pct": ols_pct,
        "rf_pct": rf_pct,
        "regime": regime_info["regime"],
        "confidence_mult": regime_info["confidence_mult"],
        "raw_pct": raw_pct,
    }

def format_intraday_forecast(result: dict) -> str:
    if "error" in result:
        return f"⚠️ {result['error']}"

    pct = result["predicted_pct"]
    symbol = result["symbol"]
    cur_price = result.get("current_price")
    pred_price = result.get("predicted_price")

    if pct > 0.05:
        arrow = "📈"
        direction = "рост"
    elif pct < -0.05:
        arrow = "📉"
        direction = "падение"
    else:
        arrow = "➡️"
        direction = "боковик"

    lines = [
        f"{arrow} <b>{symbol}:</b> прогноз <code>{pct:+.3f}%</code> ({direction})",
    ]

    if cur_price and pred_price:
        diff = pred_price - cur_price
        lines.append(
            f"💰 <code>${cur_price:,.2f}</code> → <code>${pred_price:,.2f}</code> "
            f"({'+' if diff >= 0 else ''}{diff:,.2f}$)"
        )

    inputs = result.get("inputs", {})
    ob = inputs.get("ob_imb", 0)
    if ob > 0.1:
        ob_text = "🟢 покупатели давят"
    elif ob < -0.1:
        ob_text = "🔴 продавцы давят"
    else:
        ob_text = "⚪ баланс"

    lag1 = inputs.get("ret_lag1", 0)
    if lag1 > 0.001:
        momentum = "📈 растущий"
    elif lag1 < -0.001:
        momentum = "📉 падающий"
    else:
        momentum = "➡️ нейтральный"

    fg = inputs.get("fear_greed", 0)
    fg_val = int(fg * 50 + 50)

    ret15 = inputs.get("ret_15m", 0)
    flow_br = inputs.get("flow_buy_ratio", 0)
    vol = inputs.get("realized_vol", 0)

    if ret15 > 0.0005 and ob > 0.1 and flow_br > 0.1:
        phase = "🟢 Бычий тренд"
    elif ret15 < -0.0005 and ob < -0.1 and flow_br < -0.1:
        phase = "🔴 Медвежий тренд"
    elif vol < 0.0005 and abs(ob) < 0.1:
        phase = "🟡 Накопление"
    elif vol > 0.002 and abs(flow_br) < 0.1:
        phase = "🟠 Распределение"
    else:
        phase = "⚪ Неопределённость"

    lines += [
        f"📊 Стакан: {ob_text}",
        f"⚡ Моментум: {momentum}",
        f"😱 Страх/Жадность: <code>{fg_val}/100</code>",
        f"🏷️ Фаза: {phase}",
    ]

    if inputs.get("vol_anomaly"):
        lines.append("⚠️ <b>Аномальная волатильность!</b> Повышенный риск")

    cascade = inputs.get("cascade_risk", 0)
    if cascade > 0.7:
        lines.append("🚨 <b>Высокий риск каскада ликвидаций!</b>")
    elif cascade > 0.5:
        lines.append("⚠️ Повышенный риск каскада ликвидаций")

    return "\n".join(lines)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger("test")

    for sym in ("btc", "eth"):
        r = get_intraday_forecast(sym, logger=log)
        if "error" not in r:
            print(format_intraday_forecast(r))
            print()
        else:
            print(f"{sym.upper()}: {r['error']}")
