import requests

BINANCE_BASE = "https://fapi.binance.com"

SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

TIMEOUT = 10

def get_futures_data(cfg, logger):
    result = {}
    for symbol, binance_symbol in SYMBOLS.items():
        logger.info(f"Binance Futures [{symbol}]: загрузка...")
        oi   = _fetch_open_interest(binance_symbol, symbol, logger)
        stat = _fetch_24h_stats(binance_symbol, symbol, logger)
        liq  = _fetch_ls_ratio(binance_symbol, symbol, logger)
        result[symbol] = {
            "open_interest": oi,
            "liquidations":  liq,
            "stats":         stat,
            "alert":         _build_alert(symbol, oi, liq),
        }
    return result

def _fetch_open_interest(binance_symbol, symbol, logger):
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/futures/data/openInterestHist",
            params={"symbol": binance_symbol, "period": "1d", "limit": 2},
            timeout=TIMEOUT
        )
        if resp.status_code == 451:
            raise Exception("451 геоблок")
        resp.raise_for_status()
        hist = resp.json()

        oi_usd, change_pct = None, 0
        if hist and len(hist) >= 2:
            last = float(hist[-1].get("sumOpenInterestValue", 0))
            prev = float(hist[-2].get("sumOpenInterestValue", 0))
            oi_usd     = round(last / 1e9, 3)
            change_pct = round((last - prev) / prev * 100 if prev else 0, 2)
        elif hist:
            oi_usd = round(float(hist[0].get("sumOpenInterestValue", 0)) / 1e9, 3)

        signal = (
            f"🔴 OI растёт (+{change_pct:.1f}%)"  if change_pct > 3  else
            f"🟢 OI снижается ({change_pct:.1f}%)" if change_pct < -3 else
            f"🟡 OI стабилен ({change_pct:+.1f}%)"
        )
        logger.info(f"OI [{symbol}]: ${oi_usd}B ({change_pct:+.1f}%)")
        return {"oi_usd": oi_usd, "change_pct": change_pct, "signal": signal}

    except Exception as e:
        logger.warning(f"OI endpoint 1 [{symbol}]: {e}, пробуем endpoint 2...")

    try:
        resp = requests.get(
            f"{BINANCE_BASE}/fapi/v1/openInterest",
            params={"symbol": binance_symbol},
            timeout=TIMEOUT
        )
        if resp.status_code == 451:
            raise Exception("451 геоблок")
        resp.raise_for_status()
        data = resp.json()
        oi_contracts = float(data.get("openInterest", 0))

        price_resp = requests.get(
            f"{BINANCE_BASE}/fapi/v1/ticker/price",
            params={"symbol": binance_symbol},
            timeout=TIMEOUT
        )
        price = float(price_resp.json().get("price", 0)) if price_resp.ok else 0
        oi_usd = round(oi_contracts * price / 1e9, 3) if price else None

        logger.info(f"OI [{symbol}] endpoint2: ${oi_usd}B")
        return {
            "oi_usd":      oi_usd,
            "change_pct":  None,
            "signal": "🟡 OI стабилен",
        }

    except Exception as e:
        logger.error(f"OI [{symbol}] все endpoints недоступны: {e}")
        return None

def _fetch_24h_stats(binance_symbol, symbol, logger):
    volume_usd   = None
    funding_rate = None

    try:
        resp = requests.get(
            f"{BINANCE_BASE}/fapi/v1/ticker/24hr",
            params={"symbol": binance_symbol},
            timeout=TIMEOUT
        )
        if resp.status_code != 451:
            resp.raise_for_status()
            data = resp.json()
            volume_usd = round(float(data.get("quoteVolume", 0)) / 1e9, 2)
    except Exception as e:
        logger.warning(f"Volume [{symbol}]: {e}")

    for url in [
        f"{BINANCE_BASE}/fapi/v1/premiumIndex",
        f"{BINANCE_BASE}/fapi/v1/fundingRate",
    ]:
        try:
            params = {"symbol": binance_symbol}
            if "fundingRate" in url:
                params["limit"] = 1
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code == 451:
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                data = data[-1] if data else {}
            funding_rate = round(float(data.get("lastFundingRate", data.get("fundingRate", 0))) * 100, 4)
            break
        except Exception:
            continue

    if volume_usd is None and funding_rate is None:
        return None

    funding_signal = (
        "🔴 Высокий funding (лонги переплачивают)"       if (funding_rate or 0) > 0.05  else
        "🟢 Отрицательный funding (шорты переплачивают)" if (funding_rate or 0) < -0.01 else
        "🟡 Нормальный funding"
    )

    logger.info(f"Stats [{symbol}]: vol=${volume_usd}B, funding={funding_rate}%")
    return {
        "volume_24h_usd": volume_usd,
        "funding_rate":   funding_rate,
        "funding_signal": funding_signal,
    }

def _fetch_ls_ratio(binance_symbol, symbol, logger):
    for period in ["1h", "4h"]:
        try:
            resp = requests.get(
                f"{BINANCE_BASE}/futures/data/globalLongShortAccountRatio",
                params={"symbol": binance_symbol, "period": period, "limit": 1},
                timeout=TIMEOUT
            )
            if resp.status_code == 451:
                logger.warning(f"L/S [{symbol}]: 451 геоблок, пробуем topLongShortAccountRatio...")
                break
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue

            last        = data[-1]
            long_ratio  = round(float(last.get("longAccount", 0)) * 100, 1)
            short_ratio = round(float(last.get("shortAccount", 0)) * 100, 1)

            signal = (
                "🔴 Много лонгов — риск каскадных ликвидаций" if long_ratio > 60 else
                "🟢 Много шортов — риск шорт-сквиза"          if short_ratio > 60 else
                "🟡 Позиции сбалансированы"
            )
            logger.info(f"L/S [{symbol}]: long={long_ratio}%, short={short_ratio}%")
            return {"long_ratio": long_ratio, "short_ratio": short_ratio, "signal": signal}

        except Exception as e:
            logger.warning(f"L/S [{symbol}] period={period}: {e}")

    try:
        resp = requests.get(
            f"{BINANCE_BASE}/futures/data/topLongShortAccountRatio",
            params={"symbol": binance_symbol, "period": "1h", "limit": 1},
            timeout=TIMEOUT
        )
        if resp.status_code == 451:
            logger.error(f"L/S [{symbol}]: все endpoints заблокированы (451)")
            return None
        resp.raise_for_status()
        data = resp.json()
        if data:
            last        = data[-1]
            long_ratio  = round(float(last.get("longAccount", 0)) * 100, 1)
            short_ratio = round(float(last.get("shortAccount", 0)) * 100, 1)
            signal = (
                "🔴 Много лонгов (топ трейдеры)" if long_ratio > 60 else
                "🟢 Много шортов (топ трейдеры)" if short_ratio > 60 else
                "🟡 Позиции сбалансированы (топ трейдеры)"
            )
            logger.info(f"L/S top [{symbol}]: long={long_ratio}%, short={short_ratio}%")
            return {"long_ratio": long_ratio, "short_ratio": short_ratio, "signal": signal}
    except Exception as e:
        logger.error(f"L/S fallback [{symbol}]: {e}")

    return None

def _build_alert(symbol, oi, liq):
    if not oi or not liq:
        return None
    change = oi.get("change_pct") or 0
    if change > 3 and liq.get("long_ratio", 0) > 60:
        return f"⚠️ [{symbol}] Рост OI + перевес лонгов → риск коррекции!"
    if change < -3 and liq.get("short_ratio", 0) > 60:
        return f"💡 [{symbol}] Снижение OI + перевес шортов → возможный отскок"
    return None
