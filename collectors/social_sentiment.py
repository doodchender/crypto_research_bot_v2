import time
import logging
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed

_finbert_pipeline = None
_finbert_loaded = False

REDDIT_SUBREDDITS = [
    "cryptocurrency", "bitcoin", "ethtrader", "CryptoMarkets",
    "btc", "ethereum", "defi", "altcoins", "SatoshiStreetBets",
    "CryptoMoonShots", "BitcoinMarkets", "ethfinance",
    "binance", "solana",
]

CRYPTO_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
    "defi", "nft", "altcoin", "trading", "bull", "bear", "pump",
    "dump", "hodl", "moon", "dip", "whale", "binance", "coinbase",
    "solana", "sol", "bnb", "xrp", "market", "price", "buy", "sell",
}

_prev_sentiment = {"value": 0.0, "ts": 0}
_trends_cache = {"interest": 50.0, "ts": 0}

def _load_finbert():
    global _finbert_pipeline, _finbert_loaded
    if _finbert_loaded:
        return _finbert_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        import os

        local_model = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "finbert_model")
        if os.path.exists(local_model):
            _finbert_pipeline = hf_pipeline(
                "sentiment-analysis", model=local_model,
                tokenizer=local_model, truncation=True, max_length=512
            )
            print(f"FinBERT loaded OK (from {local_model})")
        else:
            _finbert_pipeline = hf_pipeline(
                "sentiment-analysis", model="ProsusAI/finbert",
                truncation=True, max_length=512
            )
            print("FinBERT loaded OK (from ProsusAI/finbert)")
    except Exception as e:
        print(f"FinBERT load error: {e}")
        _finbert_pipeline = None

    _finbert_loaded = True
    return _finbert_pipeline

def _score_text(text: str) -> float:
    pipe = _load_finbert()
    if pipe is None:
        return 0.0

    try:
        result = pipe(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0.0
    except Exception:
        return 0.0

def _fetch_subreddit(sub: str) -> list[dict]:
    import requests
    url = f"https://www.reddit.com/r/{sub}/new/.rss?limit=25"
    try:
        r = requests.get(url, headers={"User-Agent": "CryptoBot/1.0"}, timeout=8)
        if r.status_code != 200:
            return []
        feed = feedparser.parse(r.text)
        posts = []
        for entry in feed.entries:
            title = entry.get("title", "")
            title_lower = title.lower()
            if any(kw in title_lower for kw in CRYPTO_KEYWORDS):
                posts.append({
                    "title": title,
                    "source": f"r/{sub}",
                    "published": entry.get("published", ""),
                })
        return posts
    except Exception:
        return []

def _fetch_google_trends(logger: logging.Logger) -> float:
    global _trends_cache

    if time.time() - _trends_cache["ts"] < 300:
        return _trends_cache["interest"]

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=180, timeout=(5, 10))
        pytrends.build_payload(["bitcoin"], timeframe="now 1-H")
        data = pytrends.interest_over_time()
        if not data.empty:
            interest = float(data["bitcoin"].iloc[-1])
            _trends_cache["interest"] = interest
            _trends_cache["ts"] = time.time()
            return interest
    except Exception as e:
        if logger:
            logger.debug(f"Google Trends error: {e}")

    return _trends_cache["interest"]

def get_social_sentiment(logger: logging.Logger) -> dict:
    global _prev_sentiment

    if logger:
        logger.info("Social sentiment: fetching from Reddit (14 subs)...")

    all_posts = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(_fetch_subreddit, sub): sub
                   for sub in REDDIT_SUBREDDITS}
        for future in as_completed(futures):
            try:
                posts = future.result()
                all_posts.extend(posts)
            except Exception:
                pass

    if logger:
        logger.info(f"Social sentiment: {len(all_posts)} posts from Reddit")

    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique_posts.append(p)

    scored = []
    for post in unique_posts:
        score = _score_text(post["title"])
        scored.append({**post, "sentiment": score})

    if scored:
        sentiments = [s["sentiment"] for s in scored]
        avg = sum(sentiments) / len(sentiments)
        volume = len(scored)
    else:
        avg = 0.0
        volume = 0

    change = avg - _prev_sentiment["value"]
    _prev_sentiment = {"value": avg, "ts": time.time()}

    trends = _fetch_google_trends(logger)

    if avg > 0.15:
        signal = "🟢 Позитивный фон в соцсетях"
    elif avg < -0.15:
        signal = "🔴 Негативный фон в соцсетях"
    else:
        signal = "🟡 Нейтральный фон в соцсетях"

    sorted_posts = sorted(scored, key=lambda x: x["sentiment"], reverse=True)
    top_positive = sorted_posts[:3] if scored else []
    top_negative = sorted_posts[-3:][::-1] if scored else []

    if logger:
        logger.info(
            f"Social sentiment: avg={avg:+.3f}, posts={volume}, "
            f"change={change:+.3f}, trends={trends:.0f}, signal={signal}"
        )

    return {
        "avg_sentiment": avg,
        "count": volume,
        "signal": signal,
        "model": "FinBERT",
        "top_positive": top_positive,
        "top_negative": top_negative,
        "social_sentiment": avg,
        "social_volume": volume,
        "social_sentiment_change": change,
        "trends_interest": trends,
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("test")
    result = get_social_sentiment(log)
    print(f"\nSentiment: {result['avg_sentiment']:+.3f}")
    print(f"Posts: {result['social_volume']}")
    print(f"Change: {result['social_sentiment_change']:+.3f}")
    print(f"Trends: {result['trends_interest']}")
    print(f"Signal: {result['signal']}")
    if result["top_positive"]:
        print(f"\nTop positive:")
        for p in result["top_positive"]:
            print(f"  [{p['source']}] ({p['sentiment']:+.2f}) {p['title'][:70]}")
    if result["top_negative"]:
        print(f"\nTop negative:")
        for p in result["top_negative"]:
            print(f"  [{p['source']}] ({p['sentiment']:+.2f}) {p['title'][:70]}")
