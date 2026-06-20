import math
import logging
from typing import Optional, List
from collections import deque

import numpy as np

logger = logging.getLogger("volatility")

MIN_OBS_FOR_EGARCH = 100
REFIT_INTERVAL = 288
ROLLING_WINDOW = 30

class VolatilityEstimator:

    def __init__(self, asset: str = "BTC"):
        self.asset = asset
        self.returns: deque = deque(maxlen=5000)
        self.model_params: Optional[dict] = None
        self.cond_var: float = 0.0
        self.updates_since_fit: int = 0
        self.fitted: bool = False

    def fit(self, returns_series: List[float]) -> bool:
        self.returns.extend(returns_series)

        if len(self.returns) < MIN_OBS_FOR_EGARCH:
            logger.info(f"[{self.asset}] EGARCH: мало данных ({len(self.returns)}/{MIN_OBS_FOR_EGARCH}), fallback на rolling std")
            self.fitted = False
            return False

        try:
            from arch import arch_model

            rets = np.array(self.returns) * 100

            model = arch_model(rets, vol="EGARCH", p=1, q=1, dist="t",
                               mean="Zero", rescale=False)
            result = model.fit(disp="off", show_warning=False)

            if result.convergence_flag != 0:
                logger.warning(f"[{self.asset}] EGARCH не сошёлся (flag={result.convergence_flag}), fallback")
                self.fitted = False
                return False

            params = result.params
            self.model_params = {
                "omega": float(params.get("omega", 0)),
                "alpha": float(params.get("alpha[1]", 0)),
                "gamma": float(params.get("gamma[1]", 0)),
                "beta": float(params.get("beta[1]", 0)),
                "nu": float(params.get("nu", 5)),
            }

            self.cond_var = float(result.conditional_volatility.iloc[-1] ** 2) / 10000
            self.updates_since_fit = 0
            self.fitted = True

            logger.info(
                f"[{self.asset}] EGARCH обучен: omega={self.model_params['omega']:.4f}, "
                f"alpha={self.model_params['alpha']:.4f}, beta={self.model_params['beta']:.4f}, "
                f"gamma={self.model_params['gamma']:.4f}, nu={self.model_params['nu']:.2f}, "
                f"cond_vol={math.sqrt(self.cond_var):.6f}"
            )
            return True

        except Exception as e:
            logger.error(f"[{self.asset}] EGARCH fit error: {e}")
            self.fitted = False
            return False

    def update(self, new_return: float) -> float:
        self.returns.append(new_return)
        self.updates_since_fit += 1

        if self.updates_since_fit >= REFIT_INTERVAL:
            self.fit(list(self.returns))

        if not self.fitted or not self.model_params:
            return self._rolling_std()

        p = self.model_params
        prev_var = max(self.cond_var, 1e-20)
        prev_vol = math.sqrt(prev_var)

        z = new_return / (prev_vol + 1e-15)

        nu = p["nu"]
        if nu > 2:
            from math import gamma as gamma_func
            e_abs_z = math.sqrt((nu - 2) / math.pi) * gamma_func((nu - 1) / 2) / gamma_func(nu / 2)
        else:
            e_abs_z = math.sqrt(2 / math.pi)

        log_var = (p["omega"]
                   + p["alpha"] * (abs(z) - e_abs_z)
                   + p["gamma"] * z
                   + p["beta"] * math.log(prev_var + 1e-20))

        log_var = max(min(log_var, 0), -30)
        self.cond_var = math.exp(log_var) / 10000

        return math.sqrt(self.cond_var)

    def get_vol_forecast(self) -> float:
        if self.fitted and self.cond_var > 0:
            return math.sqrt(self.cond_var)
        return self._rolling_std()

    def is_anomaly(self) -> bool:
        if not self.fitted or not self.model_params:
            vol = self._rolling_std()
            if vol <= 0 or len(self.returns) < 10:
                return False
            last_ret = self.returns[-1] if self.returns else 0
            return abs(last_ret) > 3 * vol

        nu = self.model_params["nu"]
        threshold = self.get_student_t_threshold(0.99)
        vol = math.sqrt(self.cond_var) if self.cond_var > 0 else self._rolling_std()
        last_ret = self.returns[-1] if self.returns else 0

        return abs(last_ret) > threshold * vol

    def get_student_t_threshold(self, p: float = 0.99) -> float:
        if not self.fitted or not self.model_params:
            if p == 0.99:
                return 2.33
            elif p == 0.975:
                return 1.96
            return 2.33

        nu = self.model_params["nu"]
        try:
            from scipy.stats import t as t_dist
            return float(t_dist.ppf(p, df=nu))
        except ImportError:
            if p == 0.99:
                return 4.54 if nu <= 3.5 else 3.75 if nu <= 5 else 2.33
            elif p == 0.975:
                return 3.18 if nu <= 3.5 else 2.57 if nu <= 5 else 1.96
            return 2.33

    def get_whale_threshold(self) -> float:
        return self.get_student_t_threshold(0.975)

    def _rolling_std(self) -> float:
        if len(self.returns) < 3:
            return 0.0
        recent = list(self.returns)[-ROLLING_WINDOW:]
        mean_r = sum(recent) / len(recent)
        return math.sqrt(sum((r - mean_r) ** 2 for r in recent) / len(recent))

    def get_status(self) -> dict:
        return {
            "asset": self.asset,
            "fitted": self.fitted,
            "observations": len(self.returns),
            "updates_since_fit": self.updates_since_fit,
            "cond_vol": math.sqrt(self.cond_var) if self.cond_var > 0 else 0,
            "nu": self.model_params.get("nu") if self.model_params else None,
            "anomaly": self.is_anomaly(),
        }
