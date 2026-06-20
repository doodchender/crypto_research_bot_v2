import math
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RiskConfig:
    half_kelly: float = 0.125
    max_position_pct: float = 0.20

    whale_boost: float = 1.2
    sentiment_boost: float = 1.1
    both_boost: float = 1.3

    cascade_blocker: float = 0.5
    cascade_warning: float = 0.4
    cascade_reduce_mult: float = 0.5

    dd_level_1: float = -0.015
    dd_level_2: float = -0.025
    dd_level_3: float = -0.035

    dd_level_1_mult: float = 0.5
    dd_level_2_mult: float = 0.25

    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    min_sl_pct: float = 0.10
    max_sl_pct: float = 0.80

    position_timeout_candles: int = 6

@dataclass
class DailyState:
    date: str
    start_capital: float
    current_capital: float
    realized_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    level_reached: int = 0

    @property
    def daily_drawdown(self) -> float:
        if self.start_capital <= 0:
            return 0
        return (self.current_capital - self.start_capital) / self.start_capital

class RiskManager:

    def __init__(self, initial_capital: float = 1000.0, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily: Optional[DailyState] = None
        self._stopped_today = False

    def start_new_day(self, date: str = None):
        if date is None:
            date = time.strftime("%Y-%m-%d")
        self.daily = DailyState(
            date=date,
            start_capital=self.current_capital,
            current_capital=self.current_capital,
        )
        self._stopped_today = False

    def register_trade_pnl(self, pnl_usd: float):
        self.current_capital += pnl_usd
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        if self.daily is not None:
            self.daily.current_capital = self.current_capital
            self.daily.realized_pnl += pnl_usd
            self.daily.trades_count += 1
            if pnl_usd > 0:
                self.daily.wins += 1
            else:
                self.daily.losses += 1

    def get_size_multiplier_from_drawdown(self) -> tuple[float, int]:
        if self.daily is None:
            return 1.0, 0
        dd = self.daily.daily_drawdown
        cfg = self.config

        if dd <= cfg.dd_level_3:
            return 0.0, 3
        elif dd <= cfg.dd_level_2:
            return cfg.dd_level_2_mult, 2
        elif dd <= cfg.dd_level_1:
            return cfg.dd_level_1_mult, 1
        return 1.0, 0

    def evaluate_entry(
        self,
        predicted_pct: float,
        side: str,
        features: dict,
        current_capital: Optional[float] = None,
        atr_pct: Optional[float] = None,
    ) -> dict:
        cfg = self.config
        cap = current_capital if current_capital is not None else self.current_capital

        result = {
            "allow": False,
            "reason": "",
            "size_usd": 0.0,
            "size_pct": 0.0,
            "sl_pct": 0.0,
            "tp_pct": 0.0,
            "level": 0,
            "boost_multiplier": 1.0,
        }

        dd_mult, level = self.get_size_multiplier_from_drawdown()
        result["level"] = level
        if level >= 3:
            result["reason"] = "day_stopped"
            return result

        cascade = float(features.get("cascade_risk") or 0)
        if cascade >= cfg.cascade_blocker:
            result["reason"] = f"cascade_high ({cascade:.2f})"
            return result

        if features.get("vol_anomaly"):
            result["reason"] = "vol_anomaly"
            return result

        direction_sign = 1 if side == "long" else -1
        whale_imb = float(features.get("whale_imb") or 0)
        sentiment = float(features.get("social_sentiment") or 0)

        whale_ok = (whale_imb * direction_sign) > 0.1
        sent_ok = (sentiment * direction_sign) > 0.1

        boost = 1.0
        if whale_ok and sent_ok:
            boost = cfg.both_boost
        elif whale_ok:
            boost = cfg.whale_boost
        elif sent_ok:
            boost = cfg.sentiment_boost

        cascade_mult = 1.0
        if cascade >= cfg.cascade_warning:
            cascade_mult = cfg.cascade_reduce_mult

        size_pct = cfg.half_kelly * dd_mult * boost * cascade_mult
        size_pct = min(size_pct, cfg.max_position_pct)
        size_usd = cap * size_pct

        if atr_pct is None:
            atr_pct = float(features.get("realized_vol") or 0.001) * 100
        atr_pct = max(atr_pct, cfg.min_sl_pct)

        sl_pct = atr_pct * cfg.sl_atr_mult / 100
        sl_pct = min(max(sl_pct, cfg.min_sl_pct / 100), cfg.max_sl_pct / 100)

        tp_pct = atr_pct * cfg.tp_atr_mult / 100

        if cascade >= cfg.cascade_warning:
            tp_pct *= 1.3

        result.update({
            "allow": True,
            "reason": "OK",
            "size_usd": round(size_usd, 2),
            "size_pct": round(size_pct, 4),
            "sl_pct": round(sl_pct, 5),
            "tp_pct": round(tp_pct, 5),
            "boost_multiplier": round(boost * cascade_mult * dd_mult, 3),
        })
        return result

    def get_status(self) -> dict:
        dd_mult, level = self.get_size_multiplier_from_drawdown()
        state = {
            "capital": round(self.current_capital, 2),
            "peak": round(self.peak_capital, 2),
            "total_return_pct": round((self.current_capital - self.initial_capital) /
                                       self.initial_capital * 100, 2),
            "level": level,
            "size_mult": dd_mult,
        }
        if self.daily:
            state.update({
                "daily_pnl": round(self.daily.realized_pnl, 2),
                "daily_dd_pct": round(self.daily.daily_drawdown * 100, 2),
                "trades_today": self.daily.trades_count,
                "win_rate_today": (round(self.daily.wins / self.daily.trades_count * 100, 1)
                                   if self.daily.trades_count > 0 else 0),
            })
        return state
