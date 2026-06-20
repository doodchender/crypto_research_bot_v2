import json
import math
import sqlite3
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

DB_6M_PATH = Path(r"C:\Users\HYPERPC\OneDrive\Рабочий стол\work\предикты\intraday_6m.db")
OUT_FILE   = Path("data/intraday_coefficients.json")
RF_DIR     = Path("data/rf_models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | train6m | %(levelname)s | %(message)s",
)
log = logging.getLogger("train6m")

def load_6m():
    conn = sqlite3.connect(DB_6M_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM intraday_snapshots
        WHERE btc_price IS NOT NULL AND eth_price IS NOT NULL
          AND btc_ret_5m IS NOT NULL AND eth_ret_5m IS NOT NULL
        ORDER BY timestamp
    """).fetchall()
    conn.close()
    log.info(f"Загружено {len(rows)} строк из 6m БД")
    return [dict(r) for r in rows]

def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def ema_series(vals, span):
    alpha = 2 / (span + 1)
    e = vals[0]
    result = [e]
    for v in vals[1:]:
        e = alpha * v + (1 - alpha) * e
        result.append(e)
    return result

def build_features(data, prefix):
    N = len(data)
    features, targets = [], []

    prices_all = [r.get(f"{prefix}_price") or 0 for r in data]
    vols_all   = [r.get(f"{prefix}_volume_5m") or 0 for r in data]

    ema5_all  = ema_series(prices_all, 5)
    ema9_all  = ema_series(prices_all, 9)
    ema12_all = ema_series(prices_all, 12)
    ema20_all = ema_series(prices_all, 20)
    ema21_all = ema_series(prices_all, 21)
    ema26_all = ema_series(prices_all, 26)
    ema50_all = ema_series(prices_all, 50)
    ema200_all= ema_series(prices_all, 200)

    other = "eth" if prefix == "btc" else "btc"
    other_prices = [r.get(f"{other}_price") or 0 for r in data]

    WARMUP = 200

    for i in range(WARMUP, N - 1):
        row = data[i]
        cur_price = prices_all[i]
        if cur_price <= 0:
            continue

        target = row.get(f"{prefix}_ret_5m")
        if target is None or not math.isfinite(target):
            continue

        feat = {}
        ts = row.get("timestamp", "")

        ob = row.get(f"{prefix}_ob_imb") or 0.5
        feat["ob_imb"] = ob * 2 - 1

        vol_5m = vols_all[i]
        feat["log_volume"] = math.log(vol_5m + 1)

        vol_window = vols_all[max(0, i-12):i]
        avg_vol = sum(vol_window) / len(vol_window) if vol_window else vol_5m
        feat["rel_volume"] = vol_5m / (avg_vol + 1e-10)

        tb = row.get(f"{prefix}_taker_buy_vol") or 0
        ts_vol = row.get(f"{prefix}_taker_sell_vol") or 0
        feat["taker_imb"] = (tb - ts_vol) / (tb + ts_vol + 1e-10)

        feat["spread_pct"] = row.get(f"{prefix}_spread_pct") or 0
        feat["funding"]    = row.get(f"{prefix}_funding") or 0

        oi_cur  = row.get(f"{prefix}_oi") or 0
        oi_prev = data[i-1].get(f"{prefix}_oi") or 0
        feat["oi_change_pct"] = (oi_cur - oi_prev) / (oi_prev + 1e-10) * 100 if oi_prev > 0 else 0

        feat["num_trades_norm"] = math.log((row.get(f"{prefix}_num_trades") or 0) + 1)

        prev_price = prices_all[i-1] if prices_all[i-1] > 0 else cur_price
        prev_vol   = vols_all[i-1]
        ret_prev   = math.log(cur_price / prev_price) if prev_price > 0 else 0
        feat["kyle_lambda"] = abs(ret_prev) / math.sqrt(prev_vol + 1)

        kl_vals = []
        for k in range(max(1, i-12), i):
            rp = prices_all[k-1]
            rc = prices_all[k]
            rv = vols_all[k]
            if rp > 0 and rc > 0:
                kl_vals.append(abs(math.log(rc / rp)) / math.sqrt(rv + 1))
        feat["kyle_lambda_roll"] = sum(kl_vals) / len(kl_vals) if kl_vals else feat["kyle_lambda"]

        for lag in range(1, 13):
            if i >= lag:
                p_cur = prices_all[i - lag + 1] if prices_all[i - lag + 1] > 0 else 1
                p_lag = prices_all[i - lag] if prices_all[i - lag] > 0 else 1
                r = math.log(p_cur / p_lag) if p_cur > 0 and p_lag > 0 else 0
                feat[f"ret_lag{lag}"] = r
            else:
                feat[f"ret_lag{lag}"] = 0.0

        p3  = prices_all[i-3]  if i >= 3  and prices_all[i-3]  > 0 else cur_price
        p6  = prices_all[i-6]  if i >= 6  and prices_all[i-6]  > 0 else cur_price
        p12 = prices_all[i-12] if i >= 12 and prices_all[i-12] > 0 else cur_price
        feat["ret_15m"] = math.log(cur_price / p3)  if cur_price > 0 and p3  > 0 else 0
        feat["ret_30m"] = math.log(cur_price / p6)  if cur_price > 0 and p6  > 0 else 0
        feat["ret_1h"]  = math.log(cur_price / p12) if cur_price > 0 and p12 > 0 else 0

        feat["acceleration"] = feat["ret_lag1"] - feat["ret_lag2"]

        op = other_prices[i-1] if i >= 1 and other_prices[i-1] > 0 else 1
        oc = other_prices[i]
        feat["other_ret_lag1"] = math.log(oc / op) if oc > 0 and op > 0 else 0
        feat["cross_diverg"]   = feat["ret_lag1"] - feat["other_ret_lag1"]

        for lag in [1, 2, 3]:
            if i >= lag:
                prev_ob = (data[i-lag].get(f"{prefix}_ob_imb") or 0.5) * 2 - 1
                ptb = data[i-lag].get(f"{prefix}_taker_buy_vol") or 0
                pts = data[i-lag].get(f"{prefix}_taker_sell_vol") or 0
                prev_ti = (ptb - pts) / (ptb + pts + 1e-10)
                feat[f"ob_imb_lag{lag}"] = prev_ob
                feat[f"taker_imb_lag{lag}"] = prev_ti
            else:
                feat[f"ob_imb_lag{lag}"] = 0.0
                feat[f"taker_imb_lag{lag}"] = 0.0
        feat["ob_imb_delta1"]    = feat["ob_imb"] - feat["ob_imb_lag1"]
        feat["ob_imb_delta3"]    = feat["ob_imb"] - feat["ob_imb_lag3"]
        feat["taker_imb_delta1"] = feat["taker_imb"] - feat["taker_imb_lag1"]
        feat["taker_imb_delta3"] = feat["taker_imb"] - feat["taker_imb_lag3"]

        prev_fund = data[i-1].get(f"{prefix}_funding") or 0 if i >= 1 else 0
        feat["funding_delta"] = feat["funding"] - prev_fund

        if i >= 2:
            oi_pp = data[i-2].get(f"{prefix}_oi") or 0
            oi_change_prev = (oi_prev - oi_pp) / (oi_pp + 1e-10) * 100 if oi_pp > 0 else 0
            feat["oi_accel"] = feat["oi_change_pct"] - oi_change_prev
        else:
            feat["oi_accel"] = 0.0

        atr_vals = [abs(prices_all[k] - prices_all[k-1]) for k in range(max(1, i-14), i+1)]
        atr = sum(atr_vals) / len(atr_vals) if atr_vals else cur_price * 0.001

        feat["ema_diff"]      = (ema9_all[i] - ema21_all[i]) / cur_price * 100
        feat["macd"]          = (ema12_all[i] - ema26_all[i]) / cur_price * 100
        feat["price_vs_ema5"] = (cur_price - ema5_all[i])  / atr if atr > 0 else 0
        feat["price_vs_ema20"]= (cur_price - ema20_all[i]) / atr if atr > 0 else 0
        feat["price_vs_ema50"]= (cur_price - ema50_all[i]) / atr if atr > 0 else 0
        feat["price_vs_ema200"]=(cur_price - ema200_all[i])/ atr if atr > 0 else 0

        p15 = [prices_all[k] for k in range(max(0, i-78), i+1, 3)]
        p1h = [prices_all[k] for k in range(max(0, i-312), i+1, 12)]
        if len(p15) >= 26:
            e12 = ema_series(p15, 12)[-1]
            e26 = ema_series(p15, 26)[-1]
            feat["macd_15m"] = (e12 - e26) / cur_price * 100
        else:
            feat["macd_15m"] = 0.0
        if len(p1h) >= 26:
            e12 = ema_series(p1h, 12)[-1]
            e26 = ema_series(p1h, 26)[-1]
            feat["macd_1h"] = (e12 - e26) / cur_price * 100
        else:
            feat["macd_1h"] = 0.0

        feat["mtf_momentum_strength"] = sgn(feat["macd"]) + sgn(feat["macd_15m"]) + sgn(feat["macd_1h"])

        win20 = prices_all[i-19:i+1]
        sma20 = sum(win20) / 20
        std20 = math.sqrt(sum((p - sma20)**2 for p in win20) / 20)
        feat["bb_position"] = (cur_price - sma20) / (2 * std20) if std20 > 0 else 0

        gains  = [max(0, prices_all[k] - prices_all[k-1]) for k in range(i-13, i+1)]
        losses = [max(0, prices_all[k-1] - prices_all[k]) for k in range(i-13, i+1)]
        ag = sum(gains) / 14
        al = sum(losses) / 14
        feat["rsi_proxy"] = (100 - 100 / (1 + ag / al)) / 100 if al > 0 else 1.0

        try:
            log_rets = [math.log(prices_all[k] / prices_all[k-1]) for k in range(i-29, i+1)]
            n = len(log_rets)
            mr = sum(log_rets) / n
            cs, cum = 0, []
            for r in log_rets:
                cs += r - mr
                cum.append(cs)
            R = max(cum) - min(cum)
            S = math.sqrt(sum((r - mr)**2 for r in log_rets) / n)
            feat["hurst"] = max(0.1, min(0.9, math.log(R/S) / math.log(n/2))) if S > 0 and R > 0 else 0.5
        except Exception:
            feat["hurst"] = 0.5

        signs = [sgn(prices_all[k] - prices_all[k-1]) for k in range(i-5, i+1)]
        feat["dir_persistence"] = sum(signs) / 6.0

        rets6 = [math.log(prices_all[k] / prices_all[k-1]) for k in range(i-5, i+1)]
        mr6   = sum(rets6) / 6
        feat["realized_vol"] = math.sqrt(sum((r - mr6)**2 for r in rets6) / 6)

        avg_v12 = sum(vols_all[i-12:i]) / 12 if i >= 12 else vol_5m
        last_ret = feat["ret_lag1"]
        feat["vol_conf_momentum"] = last_ret * math.log(vol_5m / (avg_v12 + 1e-10) + 1)

        lookback = min(60, i - WARMUP)
        if lookback >= 10:
            hist_rets = [math.log(prices_all[k] / prices_all[k-1]) for k in range(i-lookback, i)]
            mh = sum(hist_rets) / len(hist_rets)
            sh = math.sqrt(sum((r-mh)**2 for r in hist_rets) / len(hist_rets)) or 1e-10
            feat["momentum_zscore"] = (last_ret - mh) / sh
        else:
            feat["momentum_zscore"] = 0.0

        std_f = feat["realized_vol"] or 1e-10
        x = max(-0.999, min(0.999, last_ret / (3 * std_f)))
        try:
            feat["fisher_transform"] = 0.5 * math.log((1 + x) / (1 - x))
        except Exception:
            feat["fisher_transform"] = 0.0

        if i >= 26:
            pc = [prices_all[k] - prices_all[k-1] for k in range(i-25, i+1)]
            e1 = ema_series(pc, 13)[-1]
            e2 = ema_series([abs(v) for v in pc], 13)[-1]
            feat["tsi"] = e1 / e2 * 100 if e2 > 0 else 0.0
        else:
            feat["tsi"] = 0.0

        def path_eff(arr):
            if len(arr) < 2:
                return 0.0
            net = abs(arr[-1] - arr[0])
            tot = sum(abs(arr[k] - arr[k-1]) for k in range(1, len(arr)))
            return net / tot if tot > 0 else 0.0

        feat["price_efficiency"]   = path_eff(prices_all[i-9:i+1])
        feat["path_efficiency_30"] = path_eff(prices_all[i-29:i+1])
        feat["path_efficiency_60"] = path_eff(prices_all[i-59:i+1])

        w_prices = prices_all[i-19:i+1]
        w_vols   = vols_all[i-19:i+1]
        sv = sum(w_vols)
        vwap = sum(p*v for p,v in zip(w_prices, w_vols)) / sv if sv > 0 else cur_price
        feat["vwap_dev"] = (cur_price - vwap) / vwap * 100

        kc_mid = ema20_all[i]
        feat["keltner_pos"] = max(-2, min(2, (cur_price - kc_mid) / (2 * atr))) if atr > 0 else 0
        bb_width = 4 * std20
        kc_width = 4 * atr
        feat["momentum_squeeze"] = bb_width / kc_width if kc_width > 0 else 1.0

        if i >= 52:
            tenkan = (max(prices_all[i-8:i+1]) + min(prices_all[i-8:i+1])) / 2
            kijun  = (max(prices_all[i-25:i+1]) + min(prices_all[i-25:i+1])) / 2
            feat["ichimoku_conv"] = (tenkan - cur_price) / cur_price * 100
            feat["ichimoku_base"] = (kijun  - cur_price) / cur_price * 100
        else:
            feat["ichimoku_conv"] = 0.0
            feat["ichimoku_base"] = 0.0

        hi = row.get(f"{prefix}_high") or cur_price
        lo = row.get(f"{prefix}_low")  or cur_price
        feat["high_low_ratio"] = (hi - lo) / cur_price * 100

        feat["amihud"] = abs(ret_prev) / (prev_vol + 1) * 1e6

        cov_vals = []
        for k in range(i-5, i):
            r1 = math.log(prices_all[k] / prices_all[k-1]) if prices_all[k-1] > 0 else 0
            r2 = math.log(prices_all[k+1] / prices_all[k])  if prices_all[k]   > 0 else 0
            cov_vals.append(r1 * r2)
        cov = sum(cov_vals) / len(cov_vals) if cov_vals else 0
        feat["roll_measure"] = math.sqrt(max(0, -cov)) * 10000

        for win_name, win_size in [("6", 6), ("12", 12)]:
            rs = [math.log(prices_all[k] / prices_all[k-1]) for k in range(i-win_size+1, i+1)]
            mr_s = sum(rs) / len(rs)
            sr_s = math.sqrt(sum((r-mr_s)**2 for r in rs) / len(rs))
            feat[f"rolling_sharpe_{win_name}"] = mr_s / sr_s if sr_s > 0 else 0

        if feat["realized_vol"] > 0:
            rets_sk = [math.log(prices_all[k] / prices_all[k-1]) for k in range(i-5, i+1)]
            mr_sk = sum(rets_sk) / 6
            feat["skewness"] = sum(((r - mr_sk) / feat["realized_vol"])**3 for r in rets_sk) / 6
        else:
            feat["skewness"] = 0.0

        rets_ac = [math.log(prices_all[k] / prices_all[k-1]) for k in range(i-11, i+1)]
        mr_ac   = sum(rets_ac) / len(rets_ac)
        var_ac  = sum((r - mr_ac)**2 for r in rets_ac)
        if var_ac > 0 and len(rets_ac) >= 2:
            cov_ac = sum((rets_ac[k] - mr_ac) * (rets_ac[k-1] - mr_ac) for k in range(1, len(rets_ac)))
            feat["autocorr"] = cov_ac / var_ac
        else:
            feat["autocorr"] = 0.0

        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
            h, dw = dt.hour, dt.weekday()
            feat["hour_sin"]    = math.sin(2 * math.pi * h / 24)
            feat["hour_cos"]    = math.cos(2 * math.pi * h / 24)
            feat["dow_sin"]     = math.sin(2 * math.pi * dw / 7)
            feat["dow_cos"]     = math.cos(2 * math.pi * dw / 7)
            feat["session_eu"]  = 1.0 if 7 <= h < 16 else 0.0
            feat["session_us"]  = 1.0 if 13 <= h < 22 else 0.0
            feat["funding_proximity"] = 1.0 if h % 8 == 7 else 0.0
        except Exception:
            for k in ["hour_sin","hour_cos","dow_sin","dow_cos","session_eu","session_us","funding_proximity"]:
                feat[k] = 0.0

        r1 = feat["ret_lag1"]
        feat["ret_lag1_sq"]    = r1 * r1
        feat["ret_lag2_sq"]    = feat["ret_lag2"] ** 2
        feat["abs_ret_lag1"]   = abs(r1)
        feat["abs_ret_lag2"]   = abs(feat["ret_lag2"])
        feat["log_abs_ret_lag1"] = math.log(abs(r1) + 1e-10)
        feat["kyle_lambda_sq"] = feat["kyle_lambda"] ** 2

        feat["ret1_x_volume"]  = r1 * feat["log_volume"]
        feat["ret1_x_obimb"]   = r1 * feat["ob_imb"]
        feat["ret1_x_hurst"]   = r1 * (feat["hurst"] - 0.5)
        feat["ret1_x_bb"]      = r1 * feat["bb_position"]
        feat["macd_x_vol"]     = feat["macd"] * feat["realized_vol"]
        feat["mtf_x_ret1"]     = feat["mtf_momentum_strength"] * r1
        feat["kl_x_ret1"]      = feat["kyle_lambda"] * r1
        feat["kl_x_obimb"]     = feat["kyle_lambda"] * feat["ob_imb"]
        feat["vol_x_momentum"] = feat["realized_vol"] * r1

        features.append(feat)
        targets.append(target)

    return features, targets

FEATURE_NAMES = [
    "ob_imb", "taker_imb", "spread_pct", "funding",
    "ob_imb_lag1", "ob_imb_lag2", "ob_imb_lag3",
    "taker_imb_lag1", "taker_imb_lag2", "taker_imb_lag3",
    "ob_imb_delta1", "ob_imb_delta3", "taker_imb_delta1", "taker_imb_delta3",
    "funding_delta", "oi_change_pct", "oi_accel",
    "log_volume", "rel_volume", "num_trades_norm",
    "kyle_lambda", "kyle_lambda_roll",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5", "ret_lag6",
    "ret_lag7", "ret_lag8", "ret_lag9", "ret_lag10", "ret_lag11", "ret_lag12",
    "ret_15m", "ret_30m", "ret_1h", "acceleration",
    "other_ret_lag1", "cross_diverg",
    "ema_diff", "macd", "macd_15m", "macd_1h", "mtf_momentum_strength",
    "bb_position", "rsi_proxy", "dir_persistence",
    "price_vs_ema5", "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
    "vwap_dev", "keltner_pos", "momentum_squeeze",
    "ichimoku_conv", "ichimoku_base",
    "tsi", "fisher_transform", "vol_conf_momentum", "momentum_zscore",
    "realized_vol", "hurst", "high_low_ratio", "amihud", "roll_measure",
    "price_efficiency", "path_efficiency_30",
    "rolling_sharpe_6", "rolling_sharpe_12",
    "skewness", "autocorr",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_eu", "session_us", "funding_proximity",
    "ret_lag1_sq", "abs_ret_lag1",
    "ret1_x_obimb", "ret1_x_bb", "macd_x_vol", "kl_x_ret1",
]

def to_matrix(features, targets):
    X = np.array([[f.get(k, 0) or 0 for k in FEATURE_NAMES] for f in features], dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]

def train_models(symbol, features, targets):
    X, y_reg = to_matrix(features, targets)
    y_clf = (y_reg > 0).astype(int)

    log.info(f"{symbol.upper()}: {len(y_reg)} наблюдений, {X.shape[1]} фич")
    log.info(f"  Баланс классов: UP={y_clf.sum()} ({y_clf.mean()*100:.1f}%), DOWN={len(y_clf)-y_clf.sum()}")

    tscv = TimeSeriesSplit(n_splits=5)

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                                    min_samples_leaf=30, n_jobs=-1, random_state=42)
    reg_scores = cross_val_score(rf_reg, X, y_reg, cv=tscv, scoring="r2")
    log.info(f"{symbol.upper()} REG CV_R²={np.mean(reg_scores):.4f} (+/-{np.std(reg_scores):.4f})")
    rf_reg.fit(X, y_reg)

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                     min_samples_leaf=30, n_jobs=-1,
                                     random_state=42, class_weight="balanced")
    rf_acc  = cross_val_score(rf_clf, X, y_clf, cv=tscv, scoring="accuracy")
    rf_auc  = cross_val_score(rf_clf, X, y_clf, cv=tscv, scoring="roc_auc")
    log.info(f"{symbol.upper()} RFC CV_ACC={np.mean(rf_acc)*100:.2f}% (+/-{np.std(rf_acc)*100:.2f}%)")
    log.info(f"{symbol.upper()} RFC CV_AUC={np.mean(rf_auc):.4f} (+/-{np.std(rf_auc):.4f})")

    gbr_clf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          min_samples_leaf=30, random_state=42)
    gbr_acc = cross_val_score(gbr_clf, X, y_clf, cv=tscv, scoring="accuracy")
    gbr_auc = cross_val_score(gbr_clf, X, y_clf, cv=tscv, scoring="roc_auc")
    log.info(f"{symbol.upper()} GBC CV_ACC={np.mean(gbr_acc)*100:.2f}% (+/-{np.std(gbr_acc)*100:.2f}%)")
    log.info(f"{symbol.upper()} GBC CV_AUC={np.mean(gbr_auc):.4f} (+/-{np.std(gbr_auc):.4f})")

    from sklearn.ensemble import VotingClassifier
    ensemble = VotingClassifier(
        estimators=[("rfc", rf_clf), ("gbc", gbr_clf)],
        voting="soft", weights=[1, 1]
    )
    ens_acc = cross_val_score(ensemble, X, y_clf, cv=tscv, scoring="accuracy")
    ens_auc = cross_val_score(ensemble, X, y_clf, cv=tscv, scoring="roc_auc")
    log.info(f"{symbol.upper()} ENS CV_ACC={np.mean(ens_acc)*100:.2f}% (+/-{np.std(ens_acc)*100:.2f}%)")
    log.info(f"{symbol.upper()} ENS CV_AUC={np.mean(ens_auc):.4f} (+/-{np.std(ens_auc):.4f})")

    candidates = [
        ("RFC", rf_clf, float(np.mean(rf_acc)), float(np.mean(rf_auc))),
        ("GBC", gbr_clf, float(np.mean(gbr_acc)), float(np.mean(gbr_auc))),
        ("ENS", ensemble, float(np.mean(ens_acc)), float(np.mean(ens_auc))),
    ]
    best_name, best_clf, best_acc, best_auc = max(candidates, key=lambda x: x[3])
    log.info(f"{symbol.upper()} BEST={best_name} ACC={best_acc*100:.2f}% AUC={best_auc:.4f}")

    best_clf.fit(X, y_clf)
    train_acc = float(accuracy_score(y_clf, best_clf.predict(X)))

    if best_name == "ENS":
        imp = rf_clf.fit(X, y_clf).feature_importances_
    else:
        imp = best_clf.feature_importances_
    top = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:10]
    log.info(f"{symbol.upper()} TOP фичи: {[(n, f'{v:.4f}') for n, v in top]}")

    best_clf_fitted = best_clf
    probs = best_clf_fitted.predict_proba(X)[:, 1]
    for thr in [0.52, 0.54, 0.56, 0.58, 0.60]:
        mask = (probs > thr) | (probs < (1 - thr))
        if mask.sum() > 100:
            acc_thr = accuracy_score(y_clf[mask],
                                      (probs[mask] > 0.5).astype(int))
            log.info(f"  порог P>{thr:.0%}: ACC={acc_thr*100:.2f}%, n={mask.sum()} ({mask.sum()/len(y_clf)*100:.0f}%)")

    RF_DIR.mkdir(exist_ok=True)
    reg_path = RF_DIR / f"{symbol}_rf_model.pkl"
    clf_path = RF_DIR / f"{symbol}_clf_model.pkl"
    with open(reg_path, "wb") as f:
        pickle.dump({"model": rf_reg, "features": FEATURE_NAMES, "model_type": "RF_REG"}, f)
    with open(clf_path, "wb") as f:
        pickle.dump({"model": best_clf, "features": FEATURE_NAMES, "model_type": best_name}, f)
    log.info(f"  Сохранено: {reg_path}, {clf_path}")

    return {
        "ml": {
            "model_type": best_name,
            "cv_accuracy": best_acc,
            "cv_auc": best_auc,
            "train_accuracy": train_acc,
        }
    }

def main():
    data = load_6m()

    results = {}
    for prefix in ["btc", "eth"]:
        log.info(f"\n{'='*60}")
        log.info(f"  Строю фичи {prefix.upper()}...")
        features, targets = build_features(data, prefix)
        log.info(f"  {len(features)} observations")
        results[prefix] = train_models(prefix, features, targets)

    print("\n" + "="*60)
    print("  ИТОГИ")
    print("="*60)
    for sym in ["btc", "eth"]:
        r = results[sym]["ml"]
        print(f"\n{sym.upper()}:")
        print(f"  {r['model_type']} CV Accuracy: {r['cv_accuracy']*100:.2f}%")
        print(f"  {r['model_type']} CV AUC:      {r['cv_auc']:.4f}")
        print(f"  {r['model_type']} Train Acc:   {r['train_accuracy']*100:.2f}%")
        print(f"  (Baseline random = 50.0%, нужно > 52% чтобы зарабатывать)")

if __name__ == "__main__":
    main()
