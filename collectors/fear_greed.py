from utils import get_with_retry

def get_fear_greed(cfg, logger):
    data = get_with_retry(
        cfg["api"]["fear_greed_url"],
        {"limit": "2", "format": "json"},
        cfg, logger
    )
    if not data or "data" not in data:
        logger.error("Fear & Greed: нет данных")
        return None

    items = data["data"]
    current = items[0]
    value = int(current["value"])
    label = current["value_classification"]

    if value <= 25:
        emoji = "😱"
        zone  = "Extreme Fear"
    elif value <= 45:
        emoji = "😰"
        zone  = "Fear"
    elif value <= 55:
        emoji = "😐"
        zone  = "Neutral"
    elif value <= 75:
        emoji = "😊"
        zone  = "Greed"
    else:
        emoji = "🤑"
        zone  = "Extreme Greed"

    logger.info(f"Fear & Greed: {value} ({label})")

    return {
        "value": value,
        "label": label,
        "zone":  zone,
        "emoji": emoji,
    }
