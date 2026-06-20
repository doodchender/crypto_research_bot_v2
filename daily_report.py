import io
import logging
from datetime import datetime

import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from database import get_today_snapshots

MOSCOW_TZ = pytz.timezone("Europe/Moscow")

BTC_COLOR  = "#F7931A"
ETH_COLOR  = "#627EEA"
GRID_COLOR = "#2a2a2a"
BG_COLOR   = "#0f0f0f"
TEXT_COLOR = "#e0e0e0"

def _parse_rows(rows: list) -> dict:
    from datetime import datetime as dt
    data = {
        "ts":          [],
        "btc_price":   [], "eth_price":   [],
        "btc_atr":     [], "eth_atr":     [],
        "fear_greed":  [],
        "sentiment":   [],
        "btc_oi":      [], "eth_oi":      [],
    }
    for r in rows:
        try:
            ts = dt.strptime(r["timestamp"], "%Y-%m-%d %H:%M")
            ts = MOSCOW_TZ.localize(ts)
            data["ts"].append(ts)
            data["btc_price"].append(r.get("btc_price"))
            data["eth_price"].append(r.get("eth_price"))
            data["btc_atr"].append(r.get("btc_atr_pct"))
            data["eth_atr"].append(r.get("eth_atr_pct"))
            data["fear_greed"].append(r.get("fear_greed_value"))
            data["sentiment"].append(r.get("media_sentiment"))
            data["btc_oi"].append(r.get("btc_oi_usd"))
            data["eth_oi"].append(r.get("eth_oi_usd"))
        except Exception:
            continue
    return data

def _has_data(lst: list) -> bool:
    return any(v is not None for v in lst)

def build_chart(rows: list, date_str: str) -> io.BytesIO:
    data = _parse_rows(rows)
    ts   = data["ts"]

    n_panels = 3

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 3.5 * n_panels), facecolor=BG_COLOR)
    fig.suptitle(
        f"Дневной отчёт за {date_str}",
        fontsize=16, color=TEXT_COLOR, y=0.98, fontweight="bold"
    )

    gs = GridSpec(n_panels, 1, figure=fig, hspace=0.25)

    def _setup_ax(ax, ylabel):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=MOSCOW_TZ))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(True, color="#333333", linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    ax1 = fig.add_subplot(gs[0])
    _setup_ax(ax1, "Цена (USD)")
    ax1.set_title("BTC / ETH — цена", color=TEXT_COLOR, fontsize=10, pad=6)

    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(BG_COLOR)
    ax1_r.tick_params(colors=ETH_COLOR, labelsize=8)
    ax1_r.set_ylabel("ETH (USD)", color=ETH_COLOR, fontsize=9)

    btc_pts = [(t, v) for t, v in zip(ts, data["btc_price"]) if v is not None]
    eth_pts = [(t, v) for t, v in zip(ts, data["eth_price"]) if v is not None]

    if btc_pts:
        ax1.plot(*zip(*btc_pts), color=BTC_COLOR, linewidth=2, label="BTC", marker="o", markersize=3)
    if eth_pts:
        ax1_r.plot(*zip(*eth_pts), color=ETH_COLOR, linewidth=2, label="ETH", marker="o", markersize=3)

    lines1 = ax1.get_lines() + ax1_r.get_lines()
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, bbox_to_anchor=(1, 1), loc="lower right", fontsize=8,
               facecolor="#1a1a1a", edgecolor="#444", labelcolor=TEXT_COLOR)
    ax1.tick_params(axis="y", colors=BTC_COLOR)
    ax1.set_ylabel("BTC (USD)", color=BTC_COLOR, fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    _setup_ax(ax2, "ATR (%)")
    ax2.set_title("Волатильность ATR(14)", color=TEXT_COLOR, fontsize=10, pad=6)

    btc_atr = [(t, v) for t, v in zip(ts, data["btc_atr"]) if v is not None]
    eth_atr = [(t, v) for t, v in zip(ts, data["eth_atr"]) if v is not None]
    if btc_atr:
        ax2.plot(*zip(*btc_atr), color=BTC_COLOR, linewidth=1.5, label="BTC ATR%")
    if eth_atr:
        ax2.plot(*zip(*eth_atr), color=ETH_COLOR, linewidth=1.5, label="ETH ATR%")
    ax2.legend(bbox_to_anchor=(1, 1), loc="lower right", fontsize=8,
               facecolor="#1a1a1a", edgecolor="#444", labelcolor=TEXT_COLOR)

    ax3 = fig.add_subplot(gs[2])
    _setup_ax(ax3, "Сентимент")
    ax3.set_title("Медиа-сентимент", color=TEXT_COLOR, fontsize=10, pad=6)
    ax3.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.axhline(0.15, color="lime", linestyle=":", linewidth=0.7, alpha=0.5)
    ax3.axhline(-0.15, color="red", linestyle=":", linewidth=0.7, alpha=0.5)

    sent_pts = [(t, v) for t, v in zip(ts, data["sentiment"]) if v is not None]
    if sent_pts:
        xs, ys = zip(*sent_pts)
        ax3.plot(xs, ys, color="#00BFFF", linewidth=1.5, label="Sentiment", drawstyle="steps-post")
        ax3.fill_between(xs, ys, 0,
                         where=[y >= 0 for y in ys], alpha=0.2, color="lime")
        ax3.fill_between(xs, ys, 0,
                         where=[y < 0 for y in ys], alpha=0.2, color="red")
    ax3.legend(bbox_to_anchor=(1, 1), loc="lower right", fontsize=8,
               facecolor="#1a1a1a", edgecolor="#444", labelcolor=TEXT_COLOR)
    plt.tight_layout(rect=[0, 0, 0.97, 0.96])
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf

def build_caption(rows: list, date_str: str) -> str:
    if not rows:
        return f"📊 Дневной отчёт за {date_str}\n\nДанных нет."

    def _vals(key):
        return [r[key] for r in rows if r.get(key) is not None]

    lines = [f"<b>📊 Дневной отчёт за {date_str}</b>\n"]

    btc = _vals("btc_price")
    if btc:
        chg = ((btc[-1] - btc[0]) / btc[0] * 100) if btc[0] else 0
        arrow = "▲" if chg >= 0 else "▼"
        lines.append(
            f"<b>BTC:</b> <code>${btc[0]:,.0f}</code> → <code>${btc[-1]:,.0f}</code> "
            f"{arrow} <code>{chg:+.2f}%</code>"
        )
        lines.append(f"   Мин: <code>${min(btc):,.0f}</code> | Макс: <code>${max(btc):,.0f}</code>")
    lines.append("─────────────────")
    eth = _vals("eth_price")
    if eth:
        chg = ((eth[-1] - eth[0]) / eth[0] * 100) if eth[0] else 0
        arrow = "▲" if chg >= 0 else "▼"
        lines.append(
            f"<b>ETH:</b> <code>${eth[0]:,.0f}</code> → <code>${eth[-1]:,.0f}</code> "
            f"{arrow} <code>{chg:+.2f}%</code>"
        )
        lines.append(f"   Мин: <code>${min(eth):,.0f}</code> | Макс: <code>${max(eth):,.0f}</code>")

    lines.append("─────────────────")

    fg = _vals("fear_greed_value")
    if fg:
        lines.append(
            f"<b>Fear &amp; Greed:</b> Мин <code>{min(fg)}</code> | "
            f"Макс <code>{max(fg)}</code> | Среднее <code>{sum(fg)//len(fg)}</code>"
        )
    lines.append("─────────────────")

    btc_atr = _vals("btc_atr_pct")
    eth_atr = _vals("eth_atr_pct")

    sent = _vals("media_sentiment")
    if sent:
        avg_s = sum(sent) / len(sent)
        zone = "🟢 позитивный" if avg_s > 0.15 else "🔴 негативный" if avg_s < -0.15 else "🟡 нейтральный"
        lines.append(f"<b>Сентимент:</b> Среднее <code>{avg_s:+.3f}</code> — {zone}")

    lines.append(f"\n<i>Снимков за день: {len(rows)}</i>")
    return "\n".join(lines)

async def send_daily_report(context):
    cfg    = context.bot_data["cfg"]
    logger = context.bot_data["logger"]
    bot    = context.bot

    today    = datetime.now(MOSCOW_TZ)
    date_str = today.strftime("%d.%m.%Y")
    logger.info(f"Формирую дневной отчёт за {date_str}...")

    rows = get_today_snapshots(cfg, logger)

    recipients = cfg.get("daily_chat_ids", [])
    admin_id   = cfg.get("admin_id", 0)
    if admin_id and admin_id not in recipients:
        recipients.append(admin_id)

    if not recipients:
        logger.warning("daily_report: нет получателей")
        return

    caption = build_caption(rows, date_str)

    if len(rows) < 2:
        for chat_id in recipients:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"📊 Дневной отчёт за {date_str}\n\n⚠️ Недостаточно данных для графика (снимков: {len(rows)}).",
                    parse_mode="HTML"
                )
            except Exception as e:
                logger.error(f"daily_report send error: {e}")
        return

    try:
        chart_buf = build_chart(rows, date_str)
        for chat_id in recipients:
            try:
                chart_buf.seek(0)
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=chart_buf,
                    caption=caption,
                    parse_mode="HTML"
                )
                logger.info(f"Дневной отчёт отправлен: {chat_id}")
            except Exception as e:
                logger.error(f"daily_report send to {chat_id}: {e}")
    except Exception as e:
        logger.error(f"daily_report build error: {e}", exc_info=True)
        for chat_id in recipients:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"📊 Дневной отчёт за {date_str}\n\n{caption}\n\n⚠️ График не удалось построить.",
                    parse_mode="HTML"
                )
            except Exception:
                pass
