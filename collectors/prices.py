import yfinance as yf

TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

def get_live_prices(logger):
    result = {}
    for symbol, ticker in TICKERS.items():
        try:
            data = yf.Ticker(ticker)
            info = data.fast_info

            price     = round(float(info.last_price), 2)
            prev      = round(float(info.previous_close), 2)
            change    = round(price - prev, 2)
            change_pct = round((change / prev) * 100, 2) if prev else 0
            volume    = int(info.three_month_average_volume or 0)

            result[symbol] = {
                "price":      price,
                "prev_close": prev,
                "change":     change,
                "change_pct": change_pct,
                "volume":     volume,
                "emoji":      "🟢" if change >= 0 else "🔴",
            }
            logger.info(f"{symbol}: ${price} ({change_pct:+.2f}%)")

        except Exception as e:
            logger.error(f"Ошибка загрузки цены {symbol}: {e}")
            result[symbol] = None

    return result
