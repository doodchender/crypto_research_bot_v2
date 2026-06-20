import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}

def get_atr_signals(logger, window: int = 14):
    result = {}
    for symbol, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, period="90d", progress=False, auto_adjust=True)
            if raw.empty or len(raw) < window + 1:
                logger.warning(f"ATR [{symbol}]: недостаточно данных")
                result[symbol] = None
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            close = raw["Close"]
            high  = raw["High"]
            low   = raw["Low"]

            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)

            atr     = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
            atr_pct = atr / close * 100

            last_atr_pct = float(atr_pct.iloc[-1])
            median_atr   = float(atr_pct.median())
            p75_atr      = float(np.percentile(atr_pct.dropna(), 75))
            high_vol     = last_atr_pct > median_atr
            above_p75    = last_atr_pct > p75_atr
            log_ret      = float(np.log(close / close.shift(1)).iloc[-1])

            if above_p75:
                signal = "🔴 Очень высокая волатильность (>P75)"
            elif high_vol:
                signal = "⚠️ Высокая волатильность"
            else:
                signal = "✅ Нормальная волатильность"

            result[symbol] = {
                "atr":         round(float(atr.iloc[-1]), 2),
                "atr_pct":     round(last_atr_pct, 3),
                "median_atr":  round(median_atr, 3),
                "p75_atr":     round(p75_atr, 3),
                "high_vol":    high_vol,
                "above_p75":   above_p75,
                "log_return":  round(log_ret, 5),
                "signal":      signal,
            }
            logger.info(
                f"ATR [{symbol}]: {last_atr_pct:.2f}% "
                f"(медиана={median_atr:.2f}%, P75={p75_atr:.2f}%) "
                f"{'VERY HIGH' if above_p75 else 'HIGH' if high_vol else 'normal'}"
            )
        except Exception as e:
            logger.error(f"ATR [{symbol}]: {e}")
            result[symbol] = None
    return result
