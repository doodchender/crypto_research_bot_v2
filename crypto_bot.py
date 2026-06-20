import asyncio
import threading
import time
import logging
from datetime import time as dtime
import pytz
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes,
)

from utils import load_config, setup_logger, ensure_dirs
from database import init_db, get_today_snapshots, get_snapshots_range
from sentiment_impact_analyzer import run_update
from regression_forecast import get_forecast_regression, format_regression_forecast
from intraday_forecast import get_intraday_forecast, format_intraday_forecast
from whale_tracker import (start_whale_tracker, get_whale_summary,
                           get_orderbook_pressure, init_whale_db, set_whale_callback)
from intraday_collector import start_intraday_collector, stop_intraday_collector, get_intraday_stats, init_intraday_db
from paper_trader import PaperTrader
from flow_analyzer import start_flow_analyzer, get_flow_summary, get_flow_signal_text, init_flow_db
from hourly_tracker import run_hourly_snapshot
from daily_report import send_daily_report
from collectors.prices         import get_live_prices
from collectors.fear_greed     import get_fear_greed
from collectors.atr_signals    import get_atr_signals
from collectors.social_sentiment import get_social_sentiment as get_rss_sentiment
from collectors.coinglass_data import get_futures_data

cfg    = load_config()
logger = setup_logger("bot", cfg)
ensure_dirs(cfg)

BOT_TOKEN = cfg["bot_token"]
ADMIN_ID  = cfg.get("admin_id", 0)
MOSCOW_TZ = pytz.timezone("Europe/Moscow")

_paper_trader = None

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL = {
    "prices":     60,
    "fear_greed": 300,
    "atr":        300,
    "sentiment":  90,
    "futures":    120,
}

def _cached(key: str, fetch_fn, ttl_key: str):
    ttl = CACHE_TTL.get(ttl_key, 300)
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry and (time.time() - entry["ts"]) < ttl:
            logger.debug(f"Cache HIT: {key}")
            return entry["data"]
    logger.debug(f"Cache MISS: {key}")
    data = fetch_fn()
    with _CACHE_LOCK:
        _CACHE[key] = {"data": data, "ts": time.time()}
    return data

def _cache_age(key: str) -> str:
    entry = _CACHE.get(key)
    if not entry:
        return "только что"
    age = int(time.time() - entry["ts"])
    if age < 60:
        return f"{age}с назад"
    elif age < 3600:
        return f"{age // 60}м назад"
    else:
        return f"{age // 3600}ч назад"

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Дайджест",            callback_data="digest")],
        [InlineKeyboardButton("📊 История и аналитика",  callback_data="history_menu")],
        [InlineKeyboardButton("📈 Прогноз по новостям",  callback_data="forecast")],
    ])

def section_keyboard(action: str):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 Обновить",       callback_data=f"refresh:{action}"),
            InlineKeyboardButton("◀️ В меню",         callback_data="back"),
        ]
    ])

def _h(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _safe_get(d: dict, *keys, default="—"):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d if d is not None else default

async def _fetch_cached(key: str, fn, ttl_key: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _cached(key, fn, ttl_key)
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Crypto Research Bot</b>\n\n"
        "Мониторинг BTC и ETH: цены, волатильность, "
        "медиа-сентимент и фьючерсные сигналы.\n\n"
        "Выбери раздел:",
        parse_mode="HTML",
        reply_markup=main_keyboard(),
    )

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Главное меню:",
        reply_markup=main_keyboard(),
    )

async def cmd_paper_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pt = globals().get("_paper_trader")
    if pt is None:
        await update.message.reply_text("⚠️ Paper trader ещё не запущен")
        return
    try:
        text = pt.get_stats_text()
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

async def cmd_paper_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pt = globals().get("_paper_trader")
    if pt is None:
        await update.message.reply_text("⚠️ Paper trader ещё не запущен")
        return
    try:
        out_path = str(Path(cfg["paths"]["data"]) / "paper_report.xlsx")
        pt.generate_excel_report(out_path)
        with open(out_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename="paper_report.xlsx",
                caption="📊 Paper Trading отчёт",
            )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка генерации отчёта: {e}")

async def cmd_paper_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pt = globals().get("_paper_trader")
    if pt is None:
        await update.message.reply_text("⚠️ Paper trader ещё не запущен")
        return
    try:
        pt.reset_capital()
        await update.message.reply_text(
            f"🔄 Paper trader сброшен. Капитал: ${pt.risk.current_capital:,.2f}"
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

async def _safe_edit(query, text, **kwargs):
    try:
        await query.edit_message_text(text, **kwargs)
    except Exception:
        try:
            await query.message.reply_text(text, **kwargs)
            await query.delete_message()
        except Exception:
            await query.message.reply_text(text, **kwargs)

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action.startswith("refresh:"):
        section = action.split(":", 1)[1]
        if section == "digest":
            _CACHE.clear()
        elif section in _CACHE:
            del _CACHE[section]
        action = section

    if action == "whales":
        await _safe_edit(query, "⏳ Загружаю данные о китах...", parse_mode="HTML")
        text = await _get_whale_text(cfg, logger)
        await _safe_edit(query, text, parse_mode="HTML", reply_markup=_whale_keyboard())
        return

    if action == "flow":
        await _safe_edit(query, "⏳ Загружаю поток сделок...", parse_mode="HTML")
        flow_db = str(Path(cfg["paths"]["data"]) / "flow.db")
        text = _get_flow_text(flow_db)
        await _safe_edit(query, text, parse_mode="HTML", reply_markup=_flow_keyboard())
        return

    if action == "back":
        await _safe_edit(query, "📌 Главное меню:", reply_markup=main_keyboard())
        return

    await _safe_edit(query, "⏳ Загружаю данные...", parse_mode="HTML")

    if action == "forecast_5m":
        await _safe_edit(query, "⏳ Считаю 5-минутный прогноз...", parse_mode="HTML")
        try:
            text = await _get_forecast_5m_text()
        except Exception as e:
            logger.error(f"Forecast 5m error: {e}", exc_info=True)
            text = f"⚠️ Ошибка прогноза: <code>{_h(str(e))}</code>"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="forecast_5m")],
            [InlineKeyboardButton("◀️ В меню",  callback_data="back")],
        ])
        await _safe_edit(query, text, parse_mode="HTML", reply_markup=kb)
        return

    if action == "forecast":
        await _safe_edit(query, "⏳ Загружаю прогноз...", parse_mode="HTML")
        try:
            text = await _get_forecast_text()
        except Exception as e:
            logger.error(f"Forecast error: {e}", exc_info=True)
            text = f"⚠️ Ошибка прогноза: <code>{_h(str(e))}</code>"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="forecast")],
            [InlineKeyboardButton("◀️ В меню",  callback_data="back")],
        ])
        await _safe_edit(query, text, parse_mode="HTML", reply_markup=kb)
        return

    if action == "history_menu":
        await _safe_edit(query,
            "📊 <b>История и аналитика</b>\n\nВыбери период:",
            parse_mode="HTML",
            reply_markup=history_keyboard(),
        )
        return

    if action.startswith("history:"):
        period = action.split(":", 1)[1]
        await _safe_edit(query, "⏳ Строю график...", parse_mode="HTML")
        try:
            buf, caption, label = await _build_history_chart(period, cfg, logger)
            if buf is None:
                await _safe_edit(query,
                    caption, parse_mode="HTML",
                    reply_markup=history_action_keyboard(period)
                )
            else:
                await query.message.reply_photo(
                    photo=buf,
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=history_action_keyboard(period),
                )
                await query.delete_message()
        except Exception as e:
            logger.error(f"History error: {e}", exc_info=True)
            await _safe_edit(query,
                f"⚠️ Ошибка построения графика: <code>{_h(str(e))}</code>",
                parse_mode="HTML",
                reply_markup=history_keyboard(),
            )
        return

    builders = {
        "prices":     _get_prices_text,
        "fear_greed": _get_fear_greed_text,
        "atr":        _get_atr_text,
        "sentiment":  _get_sentiment_text,
        "futures":    _get_futures_text,
        "digest":     _get_digest_text,
        "forecast":   _get_forecast_text,
    }

    fn = builders.get(action)
    if not fn:
        await _safe_edit(query, "Неизвестная команда", reply_markup=main_keyboard())
        return

    try:
        text = await fn()
    except Exception as e:
        logger.error(f"Ошибка в {action}: {e}")
        text = f"⚠️ Ошибка при загрузке данных:\n<code>{_h(str(e))}</code>"

    if action == "digest":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить дайджест", callback_data="refresh:digest")],
            [InlineKeyboardButton("◀️ В меню", callback_data="back")],
        ])
    elif action == "forecast":
        kb = forecast_keyboard()
    else:
        kb = section_keyboard(action)

    await _safe_edit(query, text, parse_mode="HTML", reply_markup=kb)

def _b(text): return f"<b>{text}</b>"
def _c(text): return f"<code>{text}</code>"

async def _get_prices_text() -> str:
    try:
        prices = await _fetch_cached("prices", lambda: get_live_prices(logger), "prices")
        age = _cache_age("prices")
        lines = [f"<b>Цены BTC/ETH</b> ({age})\n"]
        for symbol in ("BTC", "ETH"):
            p = prices.get(symbol) if prices else None
            if p:
                price      = p["price"]
                change     = p["change"]
                change_pct = p["change_pct"]
                emoji      = p["emoji"]
                arrow = "▲" if change >= 0 else "▼"
                lines.append(
                    f"{emoji} <b>{symbol}</b>: <code>${price:,.2f}</code>\n"
                    f"   {arrow} <code>{change:+.2f}$</code> ({change_pct:+.2f}%)\n"
                )
            else:
                lines.append(f"<b>{symbol}</b>: ❌ нет данных\n")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Цены: ошибка — {e}"

async def _get_fear_greed_text() -> str:
    try:
        fg = await _fetch_cached("fear_greed", lambda: get_fear_greed(cfg, logger), "fear_greed")
        age = _cache_age("fear_greed")
        if not fg:
            return "😐 Fear &amp; Greed: ❌ нет данных"
        value = fg["value"]
        label = fg["label"]
        emoji = fg["emoji"]
        if value <= 25:
            note = "\n🟥 Рынок в панике — возможна перепроданность"
            comment = "\n\n💡 Исторически extreme fear — зона покупок для долгосрочных инвесторов. Краткосрочно возможно дальнейшее снижение, но риск/доходность смещается в пользу лонга."
        elif value <= 45:
            note = ""
            comment = "\n\n💡 Рынок осторожен. Трейдеры в ожидании — хорошее время для наблюдения, не для агрессивных позиций."
        elif value <= 55:
            note = ""
            comment = "\n\n💡 Нейтральный рынок. Направление не определено — лучше дождаться чёткого сигнала перед входом."
        elif value <= 75:
            note = ""
            comment = "\n\n💡 Рынок жадный. Осторожно с новыми лонгами — возможна коррекция. Хорошее время фиксировать часть прибыли."
        else:
            note = "\n🟩 Рынок жадный — возможна перекупленность"
            comment = "\n\n💡 Extreme greed — высокий риск разворота. Избегай FOMO-входов, рассмотри частичное закрытие лонгов."
        return (
            f"{emoji} <b>Fear &amp; Greed Index</b> ({age})\n\n"
            f"Значение: <code>{value}/100</code>\n"
            f"Зона: <b>{label}</b>{note}{comment}"
        )
    except Exception as e:
        return f"⚠️ Fear &amp; Greed: ошибка — {e}"

async def _get_atr_text() -> str:
    try:
        signals = await _fetch_cached("atr", lambda: get_atr_signals(logger), "atr")
        age = _cache_age("atr")
        lines = [f"<b>📊 ATR — Волатильность</b> ({age})\n─────────────────"]
        for symbol in ("BTC", "ETH"):
            s = signals.get(symbol) if signals else None
            if s:
                atr_pct    = s["atr_pct"]
                median_atr = s["median_atr"]
                log_ret    = s["log_return"]
                sig        = s["signal"]
                p75        = s.get("p75_atr", None)
                above_p75  = s.get("above_p75", False)
                high_vol   = s.get("high_vol", False)

                p75_line = ""
                if p75:
                    mark = "⬆️ выше P75" if above_p75 else "✅ ниже P75"
                    p75_line = f"\n   75-й перцентиль: <code>{p75:.2f}%</code> {mark}"

                if above_p75:
                    atr_comment = "\n   💡 Экстремальная волатильность — стопы шире обычного, размер позиции уменьши. Возможны резкие движения в обе стороны."
                elif high_vol:
                    atr_comment = "\n   💡 Повышенная волатильность — хорошие условия для трейдинга, но риск-менеджмент важнее обычного."
                else:
                    atr_comment = "\n   💡 Спокойный рынок — узкие стопы, меньше риск на сделку."

                lines.append(
                    f"<b>{symbol}</b>: <b>{sig}</b>\n"
                    f"   ATR(14): <code>{atr_pct:.2f}%</code> от цены\n"
                    f"   Медиана: <code>{median_atr:.2f}%</code>{p75_line}\n"
                    f"   Доходность: <code>{log_ret:+.4f}</code>\n"
                    f"   {atr_comment}\n"
                    f"─────────────────"
                )
            else:
                lines.append(f"<b>{symbol}</b>: ❌ нет данных\n")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ ATR: ошибка — {e}"

async def _get_sentiment_text() -> str:
    try:
        s = await _fetch_cached("sentiment", lambda: get_rss_sentiment(logger), "sentiment")
        age = _cache_age("sentiment")
        if not s:
            return "📰 Медиа-сентимент: ❌ нет данных"
        model = s.get("model", "?")
        avg   = s["avg_sentiment"]
        count = s["count"]
        sig   = s["signal"]
        if avg > 0.35:
            sent_comment = "\n💡 Очень позитивный фон — возможен FOMO. Осторожно с покупками на хайпе, рынок может быть перегрет."
        elif avg > 0.15:
            sent_comment = "\n💡 Позитивные новости поддерживают рост. Хороший фон для удержания лонгов, но следи за разворотом."
        elif avg < -0.35:
            sent_comment = "\n💡 Паника в СМИ — контрарианский сигнал покупки. Исторически сильный негатив в медиа совпадает с локальными дном."
        elif avg < -0.15:
            sent_comment = "\n💡 Негативный фон давит на цену. Лонги под давлением — дождись стабилизации сентимента перед входом."
        else:
            sent_comment = "\n💡 Нейтральный фон — рынок ищет направление. Цена будет двигаться по техническим уровням, не по новостям."

        lines = [
            f"<b>📰 Медиа-сентимент</b> ({age})\n─────────────────",
            f"Сигнал: {sig}",
            f"Среднее: <code>{avg:+.3f}</code> | Статей: <code>{count}</code>",
            sent_comment + "\n",
        ]
        if s.get("top_positive"):
            lines.append("🟢 <b>Позитивные заголовки:</b>")
            for a in s["top_positive"]:
                sc = a["sentiment"]
                title = a["title"][:75].replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  • [{a['source']}] ({sc:+.2f}) {title}...")
        if s.get("top_negative"):
            lines.append("\n🔴 <b>Негативные заголовки:</b>")
            for a in s["top_negative"]:
                sc = a["sentiment"]
                title = a["title"][:75].replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  • [{a['source']}] ({sc:+.2f}) {title}...")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Сентимент: ошибка — {e}"

async def _get_futures_text() -> str:
    try:
        futures = await _fetch_cached("futures", lambda: get_futures_data(cfg, logger), "futures")
        age = _cache_age("futures")
        lines = [f"<b>Фьючерсный рынок (Binance)</b> ({age})\n"]

        for symbol in ("BTC", "ETH"):
            f     = futures.get(symbol, {}) if futures else {}
            oi    = f.get("open_interest")
            liq   = f.get("liquidations")
            stat  = f.get("stats")
            alert = f.get("alert")

            lines.append(f"<b>{symbol}</b>")

            if oi:
                oi_usd = oi.get("oi_usd", "?")
                oi_sig = oi.get("signal", "")
                chg    = oi.get("change_pct")
                chg_str = f" | Δ24ч: <code>{chg:+.1f}%</code>" if chg is not None else ""
                lines.append(f"   OI: <code>${oi_usd}B</code>{chg_str}\n   <b>{oi_sig}</b>")
            else:
                lines.append("   📈 OI: ❌ нет данных")

            if stat:
                fr  = stat.get("funding_rate")
                fs  = stat.get("funding_signal", "")
                vol = stat.get("volume_24h_usd")
                if fr is not None:
                    lines.append(f"   💸 Funding: <code>{fr:+.4f}%</code> | {fs}")
                if vol is not None:
                    lines.append(f"   📊 Объём 24ч: <code>${vol}B</code>")

            if liq:
                lr = liq.get("long_ratio", 0)
                sr = liq.get("short_ratio", 0)
                ls = liq.get("signal", "")
                lines.append(f"   ⚖️ Лонги/Шорты: <code>{lr}% / {sr}%</code>\n   {ls}")
            else:
                lines.append("   ⚖️ L/S ratio: ❌ нет данных")

            if alert:
                lines.append(f"\n   {alert}")

            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Фьючерсы: ошибка — {e}"

async def _get_digest_text() -> str:
    try:
        loop = asyncio.get_running_loop()

        prices_f   = loop.run_in_executor(None, lambda: _cached("prices",     lambda: get_live_prices(logger),         "prices"))
        fg_f       = loop.run_in_executor(None, lambda: _cached("fear_greed", lambda: get_fear_greed(cfg, logger),     "fear_greed"))
        atr_f      = loop.run_in_executor(None, lambda: _cached("atr",        lambda: get_atr_signals(logger),         "atr"))
        sent_f     = loop.run_in_executor(None, lambda: _cached("sentiment",  lambda: get_rss_sentiment(logger),       "sentiment"))
        futures_f  = loop.run_in_executor(None, lambda: _cached("futures",    lambda: get_futures_data(cfg, logger),   "futures"))

        prices, fg, atr, sent, futures = await asyncio.gather(
            prices_f, fg_f, atr_f, sent_f, futures_f,
            return_exceptions=True
        )

        lines = ["<b>📋 Дайджест</b>\n"]

        lines.append("<b>💰 Цены:</b>")
        if isinstance(prices, dict):
            for symbol in ("BTC", "ETH"):
                p = prices.get(symbol)
                if p:
                    lines.append(
                        f"  {p['emoji']} {symbol}: <code>${p['price']:,.2f}</code> ({p['change_pct']:+.2f}%)"
                    )
        else:
            lines.append("  нет данных")

        lines.append("─────────────────")

        if isinstance(fg, dict) and fg:
            lines.append(f"{fg['emoji']} <b>Fear &amp; Greed:</b> <code>{fg['value']}/100</code> — {fg['label']}")
        else:
            lines.append("😐 <b>Fear &amp; Greed:</b> нет данных")

        lines.append("─────────────────")

        lines.append("<b>📊 Волатильность (ATR):</b>")
        if isinstance(atr, dict):
            for symbol in ("BTC", "ETH"):
                s = atr.get(symbol)
                if s:
                    p75_str = " ⬆️P75" if s.get("above_p75") else ""
                    lines.append(f"  {symbol}: {s['signal']} (<code>{s['atr_pct']:.2f}%</code>{p75_str})")
        else:
            lines.append("  нет данных")

        lines.append("─────────────────")

        if isinstance(sent, dict) and sent:
            model = sent.get("model", "?")
            lines.append(f"<b>📰 Медиа:</b> {sent['signal']} (<code>{sent['avg_sentiment']:+.3f}</code>)")
        else:
            lines.append("<b>📰 Медиа:</b> нет данных")

        lines.append("─────────────────")

        lines.append("<b>🔥 Фьючерсы:</b>")
        alerts = []
        if isinstance(futures, dict):
            for symbol in ("BTC", "ETH"):
                f  = futures.get(symbol, {})
                oi = f.get("open_interest")
                if oi:
                    chg = oi.get("change_pct")
                    chg_str = f" ({chg:+.1f}%)" if chg is not None else ""
                    lines.append(f"  {symbol} OI: <code>${oi.get('oi_usd')}B</code>{chg_str} {oi.get('signal', '')}")
                liq = f.get("liquidations")
                if liq:
                    lines.append(f"  {symbol} L/S: <code>{liq.get('long_ratio')}% / {liq.get('short_ratio')}%</code>")
                alert = f.get("alert")
                if alert:
                    alerts.append(alert)
        else:
            lines.append("  нет данных")

        if alerts:
            lines.append("\n<b>🚨 Сигналы:</b>")
            for a in alerts:
                lines.append(f"  {a}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Digest error: {e}", exc_info=True)
        return f"⚠️ Ошибка при формировании дайджеста:\n<code>{_h(str(e))}</code>"

async def send_morning_digest(context: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_ID:
        return
    _CACHE.clear()
    try:
        text = await _get_digest_text()
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=text,
            parse_mode="HTML",
        )
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text="📌 Выбери раздел:",
            reply_markup=main_keyboard(),
        )
        logger.info("Дайджест отправлен")
    except Exception as e:
        logger.error(f"Ошибка отправки дайджеста: {e}")

def forecast_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить",  callback_data="refresh:forecast")],
        [InlineKeyboardButton("◀️ В меню",    callback_data="back")],
    ])

async def _get_forecast_text() -> str:
    try:
        sent = _CACHE.get("sentiment", {}).get("data")
        if not sent:
            sent = await _fetch_cached("sentiment", lambda: get_rss_sentiment(logger), "sentiment")
        if not sent:
            return "📈 <b>Прогноз</b>\n\n⚠️ Нет данных о сентименте"

        avg   = sent.get("avg_sentiment", 0)
        model = sent.get("model", "FinBERT")

        prices_entry = _CACHE.get("prices")
        prices_data  = prices_entry.get("data") if prices_entry else None
        if not prices_data:
            raw_p = await _fetch_cached("prices", lambda: get_live_prices(logger), "prices")
            prices_data = raw_p if isinstance(raw_p, dict) else None

        fg_data = _CACHE.get("fear_greed", {}).get("data") or {}
        fg_val  = fg_data.get("value", 50)

        lines = [
            f"🎓 <b>Регрессионный прогноз</b>\n",
            f"Медиа-сентимент: <code>{avg:+.3f}</code>  |  Fear &amp; Greed: <code>{fg_val}/100</code>",
            "─────────────────",
        ]

        for sym in ("btc", "eth"):
            sym_price = prices_data.get(sym.upper(), {}).get("price") if prices_data else None
            atr_sym   = (_CACHE.get("atr", {}).get("data") or {}).get(sym.upper(), {})
            hv        = atr_sym.get("high_vol", False)

            reg      = get_forecast_regression(
                media_sentiment=avg,
                symbol=sym,
                high_vol=hv,
                fear_greed=fg_val,
                logger=logger,
            )
            reg_text = format_regression_forecast(reg, cur_price=sym_price)
            lines.append(f"\n<b>{sym.upper()}</b>\n{reg_text}")

        lines.append("\n─────────────────")
        lines.append("\n<i>Прогноз основан на исторических данных и не гарантирует результат.</i>")
        return "\n".join(lines)

    except Exception as e:
        return f"📈 <b>Прогноз</b>\n\n⚠️ Ошибка: <code>{_h(str(e))}</code>"

async def _get_forecast_5m_text() -> str:
    try:
        intraday_db = str(Path(cfg["paths"]["data"]) / "intraday.db")
        lines = ["⚡ <b>Прогноз на 5 минут</b>\n"]

        for sym in ("btc", "eth"):
            result = get_intraday_forecast(
                symbol=sym,
                db_path=intraday_db,
                logger=logger,
            )
            text = format_intraday_forecast(result)
            lines.append(text)
            lines.append("")

        lines.append("─────────────────")
        lines.append("<i>OLS-регрессия на intraday данных. Переобучай модель по мере накопления данных.</i>")
        return "\n".join(lines)

    except Exception as e:
        return f"⚡ <b>Прогноз 5 мин</b>\n\n⚠️ Ошибка: <code>{_h(str(e))}</code>"

def _whale_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить",  callback_data="whales")],
        [InlineKeyboardButton("📊 Поток сделок", callback_data="flow")],
        [InlineKeyboardButton("🏠 В меню",    callback_data="back")],
    ])

def _flow_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить",  callback_data="flow")],
        [InlineKeyboardButton("🐋 Киты",      callback_data="whales")],
        [InlineKeyboardButton("🏠 В меню",    callback_data="back")],
    ])

async def _get_whale_text(cfg: dict, logger: logging.Logger) -> str:
    db_path = str(Path(cfg.get("paths", {}).get("data", "data")) / "whale.db")

    try:
        summary = get_whale_summary(db_path, hours=1)
    except Exception as e:
        return f"🐋 <b>Киты</b>\n\n⚠️ Ошибка: <code>{_h(str(e))}</code>"

    lines = ["🐋 <b>Крупные сделки за последний час</b>\n"]

    if not summary:
        lines.append("За последний час крупных сделок не зафиксировано.")
        lines.append("\n<i>Порог: >0.5% дневного объёма биржи</i>")
        return "\n".join(lines)

    for symbol in ("BTCUSDT", "ETHUSDT"):
        sym_short = symbol.replace("USDT", "")
        data = summary.get(symbol)
        if not data:
            continue

        buy_vol  = data["buy_vol"]
        sell_vol = data["sell_vol"]
        total    = buy_vol + sell_vol
        count    = data["count"]

        if total == 0:
            continue

        buy_pct  = buy_vol / total * 100
        sell_pct = sell_vol / total * 100

        if buy_pct >= 70:
            pressure = "🟢 Сильное давление покупателей"
        elif buy_pct >= 55:
            pressure = "🟡 Умеренное давление покупателей"
        elif sell_pct >= 70:
            pressure = "🔴 Сильное давление продавцов"
        elif sell_pct >= 55:
            pressure = "🟠 Умеренное давление продавцов"
        else:
            pressure = "⚪️ Баланс сил"

        lines.append(f"\n<b>{sym_short}</b> — {count} китовых сделок")
        lines.append(f"   📈 Покупки: <code>${buy_vol:,.0f}</code> ({buy_pct:.0f}%)")
        lines.append(f"   📉 Продажи: <code>${sell_vol:,.0f}</code> ({sell_pct:.0f}%)")
        lines.append(f"   {pressure}")

        ob = get_orderbook_pressure(db_path, symbol, minutes=30)
        if ob["snapshots"] > 0:
            imb = ob["imbalance"]
            imb_pct = imb * 100
            spread  = ob["spread_pct"]
            if imb >= 0.65:
                ob_str = "🟢 Стакан перекошен в покупки"
            elif imb <= 0.35:
                ob_str = "🔴 Стакан перекошен в продажи"
            else:
                ob_str = "⚪️ Стакан сбалансирован"
            lines.append(f"   📊 Стакан: bid {imb_pct:.0f}% / ask {100-imb_pct:.0f}% — {ob_str}")
            lines.append(f"   Спред: <code>{spread:.4f}%</code>")

        trades = sorted(data.get("trades", []), key=lambda x: x["vol"], reverse=True)[:3]
        if trades:
            lines.append("   Крупнейшие:")
            for t in trades:
                arrow = "📈" if t["side"] == "buy" else "📉"
                lines.append(
                    f"   {arrow} {t['exchange'].upper()} ${t['vol']:,.0f} "
                    f"@ ${t['price']:,.2f}"
                )

    lines.append("\n─────────────────")
    lines.append("\n<i>Порог: >0.5% дневного объёма | Биржи: Binance, Bybit, OKX</i>")
    return "\n".join(lines)

async def _get_flow_text_async(flow_db: str) -> str:
    return _get_flow_text(flow_db)

def _get_flow_text(flow_db: str) -> str:
    sig_emoji = {"FOMO": "🟢🔥", "PANIC": "🔴💨", "ACTIVE": "🟡⚡", "NORMAL": "⚪️"}
    sig_desc  = {"FOMO": "Ажиотажные покупки!", "PANIC": "Массовые продажи!",
                 "ACTIVE": "Повышенная активность", "NORMAL": "Нормальная активность"}

    lines = ["📊 <b>Поток сделок (Order Flow)</b>",
             "<i>Binance + Bybit + OKX, агрегация 1 минута</i>",
             "─────────────────"]

    has_data = False
    for symbol in ("BTCUSDT", "ETHUSDT"):
        sym_short = symbol.replace("USDT", "")

        f_all = get_flow_summary(flow_db, symbol, exchange="all")
        if not f_all:
            lines.append(f"\n<b>{sym_short}</b>: ⏳ данные накапливаются...")
            continue

        has_data = True
        sig  = f_all.get("signal", "NORMAL")
        tpm  = f_all.get("trades_per_min", 0)
        avgs = f_all.get("avg_trade_size", 0)
        buy  = f_all.get("buy_ratio", 0.5) * 100
        vol  = f_all.get("total_volume", 0)
        cnt  = f_all.get("trade_count", 0)
        avg7 = f_all.get("avg7_count")
        z    = f_all.get("count_zscore", 0)
        ts   = f_all.get("timestamp", "")
        e    = sig_emoji.get(sig, "⚪️")
        d    = sig_desc.get(sig, "Норма")

        norm_str = ""
        if avg7 and avg7 > 0:
            pct = (cnt - avg7) / avg7 * 100
            norm_str = f" | vs норма: <code>{pct:+.0f}%</code> (z={z:+.1f})"

        lines.append(f"\n<b>{sym_short}</b> {e} <b>{d}</b>")
        lines.append(f"   Всего сделок: <code>{cnt}</code> (<code>{tpm:.0f}</code>/мин){norm_str}")
        lines.append(f"   Средний размер: <code>${avgs:,.0f}</code>")
        lines.append(f"   Покупки: <code>{buy:.0f}%</code> | Продажи: <code>{100-buy:.0f}%</code>")
        lines.append(f"   Объём: <code>${vol:,.0f}</code>")

        lines.append(f"   Обновлено: {ts}")

    if not has_data:
        lines += ["\n⏳ Данные пока накапливаются.",
                  "Первые данные появятся через 5 минут после запуска."]

    lines += ["\n─────────────────",
              "\n<i>FOMO=ажиотаж покупок, PANIC=паника продаж, ACTIVE=повышенная активность</i>"]
    return "\n".join(lines)

async def run_weekly_analysis(context):
    cfg_    = context.bot_data["cfg"]
    logger_ = context.bot_data["logger"]
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: run_update(cfg_, logger_))
        logger_.info("Еженедельный анализ сентимента завершён")
    except Exception as e:
        logger_.error(f"Weekly analysis error: {e}", exc_info=True)

def history_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📅 Последние 7 дней", callback_data="history:7d")],
        [InlineKeyboardButton("📅 Сегодня",          callback_data="history:today")],
        [InlineKeyboardButton("📅 Вчера",            callback_data="history:yesterday")],
        [InlineKeyboardButton("◀️ Назад",            callback_data="back")],
    ])

def history_action_keyboard(period: str):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить",     callback_data=f"history:{period}")],
        [InlineKeyboardButton("◀️ К истории",    callback_data="history_menu")],
        [InlineKeyboardButton("🏠 В меню",       callback_data="back")],
    ])

async def _build_history_chart(period: str, cfg, logger) -> tuple:
    from daily_report import build_chart, build_caption
    from database import get_today_snapshots, get_snapshots_range
    import datetime

    moscow_now = datetime.datetime.now(MOSCOW_TZ)

    if period == "today":
        rows = get_today_snapshots(cfg, logger)
        date_str = moscow_now.strftime("%d.%m.%Y")
        label = f"Сегодня ({date_str})"

    elif period == "yesterday":
        yesterday = (moscow_now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        rows = get_snapshots_range(cfg, logger, yesterday, yesterday)
        date_str = (moscow_now - datetime.timedelta(days=1)).strftime("%d.%m.%Y")
        label = f"Вчера ({date_str})"

    elif period == "7d":
        end   = moscow_now.strftime("%Y-%m-%d")
        start = (moscow_now - datetime.timedelta(days=6)).strftime("%Y-%m-%d")
        rows  = get_snapshots_range(cfg, logger, start, end)
        date_str = f"{(moscow_now - datetime.timedelta(days=6)).strftime('%d.%m')}–{moscow_now.strftime('%d.%m.%Y')}"
        label = f"7 дней ({date_str})"

    else:
        return None, "Неизвестный период", ""

    if not rows:
        return None, f"📊 <b>История: {label}</b>\n\n⚠️ Данных пока нет. Почасовой трекер накапливает данные — попробуй позже.", label

    buf     = build_chart(rows, label)
    caption = build_caption(rows, label)
    return buf, caption, label

def main():
    logger.info("Запуск Crypto Bot v2...")

    init_db(cfg, logger)

    try:
        run_update(cfg, logger)
    except Exception as e:
        logger.warning(f"Начальный анализ сентимента: {e}")

    app = Application.builder().token(BOT_TOKEN).post_init(_post_init).build()

    app.bot_data["cfg"]    = cfg
    app.bot_data["logger"] = logger

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu",  cmd_menu))
    app.add_handler(CommandHandler("paper_stats",  cmd_paper_stats))
    app.add_handler(CommandHandler("paper_report", cmd_paper_report))
    app.add_handler(CommandHandler("paper_reset",  cmd_paper_reset))
    app.add_handler(CallbackQueryHandler(on_button))

    digest_time_str = cfg.get("digest_time", "09:00")
    h, m = map(int, digest_time_str.split(":"))
    app.job_queue.run_daily(
        send_morning_digest,
        time=dtime(hour=h, minute=m, tzinfo=MOSCOW_TZ),
        name="morning_digest",
    )

    if cfg.get("hourly_snapshot_enabled", True):
        app.job_queue.run_repeating(
            run_hourly_snapshot,
            interval=3600,
            first=60,
            name="hourly_snapshot",
        )
        logger.info("Почасовой трекер запущен")

    async def _daily_report_with_menu(context):
        await send_daily_report(context)
        admin = context.bot_data["cfg"].get("admin_id", 0)
        if admin:
            try:
                await context.bot.send_message(
                    chat_id=admin,
                    text="📌 Выбери раздел:",
                    reply_markup=main_keyboard(),
                )
            except Exception:
                pass

    app.job_queue.run_daily(
        _daily_report_with_menu,
        time=dtime(hour=23, minute=59, tzinfo=MOSCOW_TZ),
        name="daily_report",
    )
    logger.info("Дневной отчёт запланирован на 23:59 МСК")

    app.job_queue.run_daily(
        run_weekly_analysis,
        time=dtime(hour=3, minute=0, tzinfo=MOSCOW_TZ),
        days=(6,),
        name="weekly_analysis",
    )
    logger.info("Еженедельный анализ запланирован на воскресенье 03:00 МСК")

    app.job_queue.run_once(
        run_weekly_analysis,
        when=30,
        name="initial_sentiment_update",
    )
    logger.info(f"Дайджест запланирован на {digest_time_str} МСК")

    whale_db    = str(Path(cfg["paths"]["data"]) / "whale.db")
    intraday_db = str(Path(cfg["paths"]["data"]) / "intraday.db")
    flow_db     = str(Path(cfg["paths"]["data"]) / "flow.db")
    init_whale_db(whale_db, logger)
    init_intraday_db(intraday_db, logger)
    init_flow_db(flow_db, logger)

    logger.info("Бот запущен. Ctrl+C для остановки.")
    app.run_polling(drop_pending_updates=True)

async def _post_init(app: Application) -> None:
    whale_db    = str(Path(cfg["paths"]["data"]) / "whale.db")
    intraday_db = str(Path(cfg["paths"]["data"]) / "intraday.db")
    flow_db     = str(Path(cfg["paths"]["data"]) / "flow.db")

    EXCHANGE_NAME = {"binance": "Binance", "bybit": "Bybit", "okx": "OKX"}

    async def _on_whale(exchange: str, symbol: str, side: str,
                        vol: float, price: float, pct: float, nickname: str):
        if not ADMIN_ID:
            return
        sym_short  = symbol.replace("USDT", "")
        exch_name  = EXCHANGE_NAME.get(exchange, exchange.upper())
        side_emoji = "📈" if side == "buy" else "📉"
        side_text  = "купил" if side == "buy" else "продал"
        text = (
            f"🐋 <b>Кит замечен на {exch_name}!</b>\n\n"
            f"👤 <b>{nickname}</b> {side_emoji} {side_text} <b>{sym_short}</b>\n"
            f"💰 Сумма: <code>${vol:,.0f}</code>\n"
            f"📊 Цена: <code>${price:,.2f}</code>\n"
            f"📈 {pct:.2f}% дневного объёма {exch_name}"
        )
        try:
            await app.bot.send_message(chat_id=ADMIN_ID, text=text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Whale notify error: {e}")

    loop = asyncio.get_running_loop()

    global _paper_trader
    paper_db = str(Path(cfg["paths"]["data"]) / "paper_trades.db")

    async def _paper_notifier(text: str):
        if ADMIN_ID:
            try:
                await app.bot.send_message(chat_id=ADMIN_ID, text=text, parse_mode="HTML")
            except Exception as e:
                logger.error(f"paper notify error: {e}")

    _paper_trader = PaperTrader(
        intraday_db=intraday_db,
        paper_db=paper_db,
        initial_capital=1000.0,
        notifier=_paper_notifier,
        logger=logger,
    )

    if cfg.get("heavy_trackers_enabled", False):
        set_whale_callback(_on_whale)
        loop.create_task(start_whale_tracker(whale_db, logger, symbols=["BTCUSDT", "ETHUSDT"]))
        loop.create_task(start_intraday_collector(intraday_db, whale_db, logger))
        loop.create_task(start_flow_analyzer(flow_db, logger))
        loop.create_task(_paper_trader.run())
        logger.info("Фоновые задачи запущены (whale / intraday / flow / paper)")
    else:
        logger.info("Режим защиты: тяжёлые трекеры отключены, прогреваю кэш...")

        async def _prewarm():
            try:
                await _fetch_cached("prices", lambda: get_live_prices(logger), "prices")
                await _fetch_cached("fear_greed", lambda: get_fear_greed(cfg, logger), "fear_greed")
                await _fetch_cached("atr", lambda: get_atr_signals(logger), "atr")
                await _fetch_cached("futures", lambda: get_futures_data(cfg, logger), "futures")
                await _fetch_cached("sentiment", lambda: get_rss_sentiment(logger), "sentiment")
                logger.info("Кэш прогрет: бот готов отвечать мгновенно")
            except Exception as e:
                logger.warning(f"Прогрев кэша: {e}")

        loop.create_task(_prewarm())

if __name__ == "__main__":
    main()
