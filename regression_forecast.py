import json
import math
import logging
import sqlite3
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta

import pytz

MOSCOW_TZ = pytz.timezone("Europe/Moscow")
COEF_FILE  = Path("data/regression_coefficients.json")

TYPICAL_VALUES = {
    "log_volume":      23.0,
    "hash_rate_th_s":  3.0e8,
    "eth_volume_usd":  8.0e9,
    "dxy":             104.0,
    "treasury_10y":    4.2,
    "gold":            2000.0,
    "fear_greed_value":50,
    "media_sentiment": 0.0,
    "media_pos":       0.0,
    "media_neg":       0.0,
    "media_lag1":      0.0,
    "media_lag2":      0.0,
    "high_vol":        0,
    "bull":            1,
    "media_x_highvol": 0.0,
    "media_pos_x_bull":0.0,
    "media_neg_x_bull":0.0,
    "media_lag1_x_highvol": 0.0,
    "media_lag2_x_highvol": 0.0,
}

_COEF_CACHE: dict = {}

def _load_coefficients() -> dict:
    global _COEF_CACHE
    if _COEF_CACHE:
        return _COEF_CACHE
    if not COEF_FILE.exists():
        raise FileNotFoundError(f"Файл коэффициентов не найден: {COEF_FILE}")
    with open(COEF_FILE, "r", encoding="utf-8") as f:
        _COEF_CACHE = json.load(f)
    return _COEF_CACHE

def reload_coefficients():
    global _COEF_CACHE
    _COEF_CACHE = {}
    return _load_coefficients()

def get_sentiment_lags(db_path: str, hours_back: int = 48) -> tuple[float, float]:
    try:
        conn = sqlite3.connect(db_path)
        now  = datetime.now(MOSCOW_TZ)

        def day_median(days_ago: int) -> float:
            dt = now - timedelta(days=days_ago)
            date_str = dt.strftime("%Y-%m-%d")
            rows = conn.execute("""
                SELECT media_sentiment FROM hourly_snapshots
                WHERE timestamp LIKE ? AND media_sentiment IS NOT NULL
            """, (f"{date_str}%",)).fetchall()
            if not rows:
                return 0.0
            vals = [r[0] for r in rows]
            vals.sort()
            return vals[len(vals) // 2]

        lag1 = day_median(1)
        lag2 = day_median(2)
        conn.close()
        return lag1, lag2
    except Exception:
        return 0.0, 0.0

def get_bull_flag(db_path: str, symbol: str) -> int:
    try:
        price_col = "btc_price" if symbol.lower() == "btc" else "eth_price"
        conn = sqlite3.connect(db_path)

        last = conn.execute(f"""
            SELECT {price_col} FROM hourly_snapshots
            WHERE {price_col} IS NOT NULL
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()

        ma200_rows = conn.execute(f"""
            SELECT {price_col} FROM hourly_snapshots
            WHERE {price_col} IS NOT NULL
            ORDER BY timestamp DESC LIMIT 4800
        """).fetchall()
        conn.close()

        if not last or not ma200_rows:
            return 1
        cur_price = last[0]
        ma200 = sum(r[0] for r in ma200_rows) / len(ma200_rows)
        return 1 if cur_price > ma200 else 0
    except Exception:
        return 1

def get_forecast_regression(
    media_sentiment: float,
    symbol: str,
    high_vol: bool = False,
    fear_greed: int = 50,
    log_volume: float = None,
    db_path: str = "data/hourly.db",
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
        model_name = sym_data.get("default_model", "M3_reduced")

    models = sym_data.get("models", {})
    if model_name not in models:
        return {"error": f"Модель {model_name} не найдена для {symbol}"}

    model = models[model_name]
    coef  = model["coef"]
    adj_r2 = model.get("adj_r2", 0)

    lag1, lag2 = get_sentiment_lags(db_path)
    bull = get_bull_flag(db_path, symbol)

    hv = 1 if high_vol else 0

    media_pos = max(media_sentiment, 0.0)
    media_neg = abs(min(media_sentiment, 0.0))

    inputs = {
        "const":               1.0,
        "log_volume":          log_volume if log_volume else TYPICAL_VALUES["log_volume"],
        "hash_rate_th_s":      TYPICAL_VALUES["hash_rate_th_s"],
        "eth_volume_usd":      TYPICAL_VALUES["eth_volume_usd"],
        "dxy":                 TYPICAL_VALUES["dxy"],
        "treasury_10y":        TYPICAL_VALUES["treasury_10y"],
        "gold":                TYPICAL_VALUES["gold"],
        "fear_greed_value":    fear_greed,
        "media_sentiment":     media_sentiment,
        "media_pos":           media_pos,
        "media_neg":           media_neg,
        "media_lag1":          lag1,
        "media_lag2":          lag2,
        "high_vol":            hv,
        "bull":                bull,
        "media_x_highvol":     media_sentiment * hv,
        "media_pos_x_bull":    media_pos * bull,
        "media_neg_x_bull":    media_neg * bull,
        "media_lag1_x_highvol": lag1 * hv,
        "media_lag2_x_highvol": lag2 * hv,
    }

    predicted_log_return = sum(
        coef[var] * inputs.get(var, 0.0)
        for var in coef
    )

    predicted_pct = (math.exp(predicted_log_return) - 1) * 100

    if logger:
        logger.debug(
            f"Regression [{symbol}/{model_name}]: "
            f"sent={media_sentiment:.3f}, fg={fear_greed}, "
            f"high_vol={high_vol}, bull={bull}, "
            f"lag1={lag1:.3f}, lag2={lag2:.3f} "
            f"→ log_ret={predicted_log_return:.4f}, pct={predicted_pct:.3f}%"
        )

    return {
        "predicted_return": predicted_log_return,
        "predicted_pct":    predicted_pct,
        "model":            model_name,
        "adj_r2":           adj_r2,
        "symbol":           symbol.upper(),
        "high_vol":         high_vol,
        "bull":             bull,
        "lag1":             lag1,
        "lag2":             lag2,
        "inputs":           inputs,
    }

def format_regression_forecast(result: dict, cur_price: float = None) -> str:
    if "error" in result:
        return f"⚠️ {result['error']}"

    pct   = result["predicted_pct"]
    model = result["model"]
    r2    = result["adj_r2"]
    bull  = result["bull"]
    lag1  = result["lag1"]

    if pct > 1.0:
        arrow   = "📈"
        prob_up = min(50 + pct * 5, 80)
    elif pct > 0.2:
        arrow   = "📈"
        prob_up = min(50 + pct * 8, 70)
    elif pct < -1.0:
        arrow   = "📉"
        prob_up = max(50 + pct * 5, 20)
    elif pct < -0.2:
        arrow   = "📉"
        prob_up = max(50 + pct * 8, 30)
    else:
        arrow   = "➡️"
        prob_up = 50.0
    prob_dn = 100 - prob_up

    dominant = max(prob_up, prob_dn)
    if dominant >= 75:
        signal_str = "💚 высокая корреляция"
    elif dominant >= 65:
        signal_str = "🟢 средняя корреляция"
    elif dominant >= 55:
        signal_str = "🟠 небольшая корреляция"
    else:
        signal_str = "🟡 почти случайно"
    if r2 >= 0.02:
        conf = "🟢 умеренная"
    elif r2 >= 0.01:
        conf = "🟠 слабая"
    else:
        conf = "🟡 очень слабая"

    lines = [
        f"{arrow} <b>Прогноз (OLS {model}):</b> <code>{pct:+.2f}%</code>",
        f"   📈 Вероятность роста: <code>{prob_up:.0f}%</code> | падения: <code>{prob_dn:.0f}%</code> ({signal_str})",
        f"   Adj. R² (точность модели, макс=1): <code>{r2:.3f}</code> | {conf}",
        f"   Тренд рынка: {'🐂 бычий' if bull else '🐻 медвежий'} | Лаг сентимента (влияние вчерашних новостей): <code>{lag1:+.3f}</code>",
    ]

    if cur_price and cur_price > 0:
        usd = cur_price * result["predicted_return"]
        lines.append(f"   💰 В долларах: <code>{usd:+,.0f}$</code>")

    return "\n".join(lines)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger("test")

    for sym in ("btc", "eth"):
        for sent in (0.2, -0.15, 0.5, 0.0):
            r = get_forecast_regression(sent, sym, high_vol=False,
                                        fear_greed=18, logger=log)
            if "error" not in r:
                print(f"{sym.upper()} sent={sent:+.2f} → {r['predicted_pct']:+.3f}% "
                      f"[{r['model']}, R²={r['adj_r2']:.3f}]")
            else:
                print(f"{sym.upper()} ERROR: {r['error']}")
