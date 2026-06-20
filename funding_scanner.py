import asyncio
import aiohttp
import csv
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

async def fetch_binance_funding(session: aiohttp.ClientSession) -> dict[str, dict]:
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            result = {}
            for item in data:
                sym = item["symbol"]
                if not sym.endswith("USDT"):
                    continue
                result[sym] = {
                    "rate": float(item.get("lastFundingRate", 0)),
                    "mark_price": float(item.get("markPrice", 0)),
                    "next_time": int(item.get("nextFundingTime", 0)),
                }
            return result
    except Exception as e:
        print(f"[Binance] ошибка: {e}")
        return {}

async def fetch_bybit_funding(session: aiohttp.ClientSession) -> dict[str, dict]:
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            items = data.get("result", {}).get("list", [])
            result = {}
            for item in items:
                sym = item["symbol"]
                if not sym.endswith("USDT"):
                    continue
                rate = item.get("fundingRate")
                if rate is None:
                    continue
                result[sym] = {
                    "rate": float(rate),
                    "mark_price": float(item.get("markPrice", 0)),
                }
            return result
    except Exception as e:
        print(f"[Bybit] ошибка: {e}")
        return {}

async def fetch_okx_funding(session: aiohttp.ClientSession) -> dict[str, dict]:
    instruments_url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
    try:
        async with session.get(instruments_url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            instruments = data.get("data", [])
    except Exception as e:
        print(f"[OKX] ошибка (instruments): {e}")
        return {}

    usdt_swaps = [
        inst["instId"] for inst in instruments
        if inst.get("settleCcy") == "USDT" or inst["instId"].endswith("-USDT-SWAP")
    ]

    result = {}
    sem = asyncio.Semaphore(10)

    async def fetch_one(inst_id: str):
        async with sem:
            url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    data = await r.json()
                    items = data.get("data", [])
                    if items:
                        rate = float(items[0].get("fundingRate", 0))
                        sym = inst_id.replace("-SWAP", "").replace("-", "")
                        result[sym] = {"rate": rate, "mark_price": 0}
            except Exception:
                pass

    tasks = [fetch_one(inst_id) for inst_id in usdt_swaps]
    await asyncio.gather(*tasks)
    return result

def analyze(
    binance: dict[str, dict],
    bybit: dict[str, dict],
    okx: dict[str, dict],
    min_abs_rate: float = 0.0005,
    min_spread: float = 0.0003,
):
    all_symbols = sorted(set(binance) | set(bybit) | set(okx))

    exchanges = {"binance": binance, "bybit": bybit, "okx": okx}

    single_opps = []
    cross_opps = []

    for sym in all_symbols:
        rates = {}
        for ex_name, ex_data in exchanges.items():
            if sym in ex_data:
                rates[ex_name] = ex_data[sym]["rate"]

        for ex_name, rate in rates.items():
            if abs(rate) >= min_abs_rate:
                ann = rate * 3 * 365 * 100
                single_opps.append({
                    "symbol": sym,
                    "exchange": ex_name,
                    "funding_rate": rate,
                    "funding_pct": rate * 100,
                    "daily_pct": rate * 3 * 100,
                    "annual_pct": ann,
                    "direction": "SHORT perp + LONG spot" if rate > 0 else "LONG perp + SHORT spot",
                })

        if len(rates) >= 2:
            ex_list = list(rates.items())
            for i in range(len(ex_list)):
                for j in range(i + 1, len(ex_list)):
                    ex_a, rate_a = ex_list[i]
                    ex_b, rate_b = ex_list[j]
                    spread = abs(rate_a - rate_b)
                    if spread >= min_spread:
                        if rate_a > rate_b:
                            short_ex, long_ex = ex_a, ex_b
                        else:
                            short_ex, long_ex = ex_b, ex_a
                        cross_opps.append({
                            "symbol": sym,
                            "short_exchange": short_ex,
                            "long_exchange": long_ex,
                            "funding_short": max(rate_a, rate_b),
                            "funding_long": min(rate_a, rate_b),
                            "spread": spread,
                            "spread_pct": spread * 100,
                            "daily_pct": spread * 3 * 100,
                            "annual_pct": spread * 3 * 365 * 100,
                        })

    single_opps.sort(key=lambda x: abs(x["funding_rate"]), reverse=True)
    cross_opps.sort(key=lambda x: x["spread"], reverse=True)

    return single_opps, cross_opps

def print_results(single_opps, cross_opps, top_n=30):
    print("\n" + "=" * 90)
    print("  SINGLE-EXCHANGE OPPORTUNITIES (спот + перп на одной бирже)")
    print("  Порог: |funding| >= 0.05%")
    print("=" * 90)

    if not single_opps:
        print("  Нет возможностей выше порога.")
    else:
        print(f"  {'Символ':<14} {'Биржа':<10} {'Funding%':>10} {'День%':>10} {'Год%':>10}  Действие")
        print("-" * 90)
        for opp in single_opps[:top_n]:
            print(
                f"  {opp['symbol']:<14} {opp['exchange']:<10} "
                f"{opp['funding_pct']:>+10.4f} {opp['daily_pct']:>+10.4f} "
                f"{opp['annual_pct']:>+10.1f}  {opp['direction']}"
            )

    print(f"\n  Всего пар с |funding| >= 0.05%: {len(single_opps)}")

    print("\n" + "=" * 90)
    print("  CROSS-EXCHANGE OPPORTUNITIES (шорт на одной + лонг на другой)")
    print("  Порог: спред funding >= 0.03%")
    print("=" * 90)

    if not cross_opps:
        print("  Нет возможностей выше порога.")
    else:
        print(
            f"  {'Символ':<14} {'Short@':<10} {'Long@':<10} "
            f"{'Спред%':>10} {'День%':>10} {'Год%':>10}"
        )
        print("-" * 90)
        for opp in cross_opps[:top_n]:
            print(
                f"  {opp['symbol']:<14} {opp['short_exchange']:<10} "
                f"{opp['long_exchange']:<10} {opp['spread_pct']:>+10.4f} "
                f"{opp['daily_pct']:>+10.4f} {opp['annual_pct']:>+10.1f}"
            )

    print(f"\n  Всего пар со спредом >= 0.03%: {len(cross_opps)}")

def save_csv(single_opps, cross_opps):
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    path_single = os.path.join(DATA_DIR, f"funding_single_{ts}.csv")
    if single_opps:
        with open(path_single, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=single_opps[0].keys())
            w.writeheader()
            w.writerows(single_opps)
        print(f"\n  Single-exchange CSV: {path_single}")

    path_cross = os.path.join(DATA_DIR, f"funding_cross_{ts}.csv")
    if cross_opps:
        with open(path_cross, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cross_opps[0].keys())
            w.writeheader()
            w.writerows(cross_opps)
        print(f"  Cross-exchange CSV:  {path_cross}")

def print_summary(binance, bybit, okx, single_opps, cross_opps):
    print("\n" + "=" * 90)
    print("  СВОДКА")
    print("=" * 90)
    print(f"  Пар на Binance:  {len(binance)}")
    print(f"  Пар на Bybit:    {len(bybit)}")
    print(f"  Пар на OKX:      {len(okx)}")
    all_syms = set(binance) | set(bybit) | set(okx)
    common = set(binance) & set(bybit) & set(okx)
    print(f"  Всего уникальных: {len(all_syms)}")
    print(f"  На всех трёх:     {len(common)}")

    for name, data in [("Binance", binance), ("Bybit", bybit), ("OKX", okx)]:
        if data:
            rates = [abs(v["rate"]) for v in data.values()]
            avg = sum(rates) / len(rates) * 100
            mx = max(rates) * 100
            max_sym = max(data, key=lambda s: abs(data[s]["rate"]))
            print(f"\n  {name}:")
            print(f"    Средний |funding|: {avg:.4f}%")
            print(f"    Макс |funding|:    {mx:.4f}% ({max_sym})")

    if single_opps:
        print(f"\n  Топ-5 single-exchange возможностей:")
        for i, opp in enumerate(single_opps[:5], 1):
            print(
                f"    {i}. {opp['symbol']} @ {opp['exchange']}: "
                f"funding={opp['funding_pct']:+.4f}%, "
                f"~{opp['annual_pct']:+.0f}% годовых"
            )

    if cross_opps:
        print(f"\n  Топ-5 cross-exchange возможностей:")
        for i, opp in enumerate(cross_opps[:5], 1):
            print(
                f"    {i}. {opp['symbol']}: short@{opp['short_exchange']} / "
                f"long@{opp['long_exchange']}, спред={opp['spread_pct']:+.4f}%, "
                f"~{opp['annual_pct']:+.0f}% годовых"
            )

async def main():
    print(f"Funding Rate Scanner — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("Загружаем данные с Binance, Bybit, OKX...")

    async with aiohttp.ClientSession() as session:
        binance, bybit, okx = await asyncio.gather(
            fetch_binance_funding(session),
            fetch_bybit_funding(session),
            fetch_okx_funding(session),
        )

    print(f"Получено: Binance={len(binance)}, Bybit={len(bybit)}, OKX={len(okx)}")

    single_opps, cross_opps = analyze(binance, bybit, okx)

    print_results(single_opps, cross_opps)
    print_summary(binance, bybit, okx, single_opps, cross_opps)
    save_csv(single_opps, cross_opps)

    print("\nГотово.")

if __name__ == "__main__":
    asyncio.run(main())
