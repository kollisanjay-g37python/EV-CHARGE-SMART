"""
src/queue_model.py
M/M/c Queuing Theory model for EV charging station wait-time estimation.

Erlang-C formula:
  P(wait > 0) = C(c, λ/μ)
  Wq = C(c, λ/μ) / (c·μ - λ)

Where:
  λ = arrival rate (vehicles/hour)
  μ = service rate (sessions/hour per port)
  c = number of charging ports (servers)
  ρ = λ / (c·μ) = server utilisation

Used as:
  1. Standalone wait-time estimator from real-time sensor data
  2. Feature generator (queue pressure) fed into ML models
  3. Capacity planning tool for operators
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    DEFAULT_ARRIVAL_RATE_PER_HOUR, DEFAULT_SERVICE_RATE_PER_HOUR,
)

logger = logging.getLogger(__name__)


@dataclass
class QueueState:
    """Snapshot of a charging station's queue state."""
    station_id: int
    station_name: str
    num_ports: int                               # c  (servers)
    arrival_rate: float                          # λ  vehicles/hr
    service_rate: float                          # μ  sessions/hr per port
    current_queue: int = 0
    current_utilization: float = 0.0

    # ─── Derived fields (computed post-init) ──────────────────────────────────
    rho: float = field(init=False)
    erlang_c: float = field(init=False)
    avg_wait_min: float = field(init=False)
    avg_queue_length: float = field(init=False)
    system_stable: bool = field(init=False)

    def __post_init__(self):
        self.rho = self._rho()
        self.system_stable = self.rho < 1.0
        self.erlang_c = self._erlang_c() if self.system_stable else 1.0
        self.avg_wait_min = self._avg_wait_min()
        self.avg_queue_length = self._avg_queue_length()

    # ─── M/M/c Formula Implementations ───────────────────────────────────────

    def _rho(self) -> float:
        """Server utilisation ρ = λ / (c·μ)."""
        denominator = self.num_ports * self.service_rate
        if denominator == 0:
            return 1.0
        return self.arrival_rate / denominator

    def _erlang_c(self) -> float:
        """
        Erlang-C probability that an arriving vehicle must wait.
        P(wait) = (cρ)^c / c! · 1/(1-ρ)
                  ─────────────────────────────────────────────
                  Σ_{n=0}^{c-1} (cρ)^n/n! + (cρ)^c/c! · 1/(1-ρ)
        """
        c = self.num_ports
        a = self.arrival_rate / self.service_rate   # offered load A = λ/μ

        # Numerator
        numerator = (a ** c / math.factorial(c)) * (1 / (1 - self.rho))

        # Denominator: sum of Poisson terms + numerator
        poisson_sum = sum(a ** n / math.factorial(n) for n in range(c))
        denominator = poisson_sum + numerator

        if denominator == 0:
            return 1.0
        return numerator / denominator

    def _avg_wait_min(self) -> float:
        """
        Average waiting time in queue Wq (minutes).
        Wq = C(c,ρ) / (c·μ - λ)
        Adjusted for current queue size when > 0.
        """
        if not self.system_stable:
            return float("inf")
        c_mu_minus_lambda = self.num_ports * self.service_rate - self.arrival_rate
        if c_mu_minus_lambda <= 0:
            return float("inf")

        # Theoretical Erlang wait (in hours → convert to minutes)
        theoretical_hr = self.erlang_c / c_mu_minus_lambda
        erlang_min = theoretical_hr * 60

        # Blend with observed queue depth
        avg_session_min = 60 / self.service_rate
        observed_min = self.current_queue * avg_session_min / max(self.num_ports, 1)
        blended = 0.5 * erlang_min + 0.5 * observed_min

        return round(max(0.0, blended), 2)

    def _avg_queue_length(self) -> float:
        """Average number of vehicles waiting Lq = λ · Wq."""
        if not self.system_stable:
            return float("inf")
        wq_hr = self.avg_wait_min / 60
        return round(self.arrival_rate * wq_hr, 3)

    # ─── Derived Metrics ──────────────────────────────────────────────────────

    @property
    def avg_system_time_min(self) -> float:
        """Total time in system W = Wq + 1/μ (minutes)."""
        return self.avg_wait_min + 60 / self.service_rate

    @property
    def avg_vehicles_in_system(self) -> float:
        """L = λ · W (Little's Law)."""
        return self.arrival_rate * self.avg_system_time_min / 60

    @property
    def throughput(self) -> float:
        """Effective throughput = min(λ, c·μ) sessions/hr."""
        return min(self.arrival_rate, self.num_ports * self.service_rate)

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "num_ports": self.num_ports,
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "rho": round(self.rho, 4),
            "erlang_c": round(self.erlang_c, 4),
            "avg_wait_min": self.avg_wait_min,
            "avg_queue_length": self.avg_queue_length,
            "avg_system_time_min": round(self.avg_system_time_min, 2),
            "throughput_per_hr": round(self.throughput, 2),
            "system_stable": self.system_stable,
            "current_queue": self.current_queue,
            "utilization_pct": round(self.rho * 100, 1),
        }


class MMcQueueModel:
    """
    M/M/c queue model for a network of EV charging stations.
    Provides:
      - Per-station wait-time estimates
      - Arrival-rate estimation from historical data
      - Capacity planning recommendations
    """

    def __init__(
        self,
        default_service_rate: float = DEFAULT_SERVICE_RATE_PER_HOUR,
        default_arrival_rate: float = DEFAULT_ARRIVAL_RATE_PER_HOUR,
    ):
        self.default_service_rate = default_service_rate
        self.default_arrival_rate = default_arrival_rate

    # ─── Single Station ───────────────────────────────────────────────────────

    def compute_wait(
        self,
        num_ports: int,
        arrival_rate: float,
        service_rate: Optional[float] = None,
        current_queue: int = 0,
        station_id: int = 0,
        station_name: str = "",
    ) -> QueueState:
        """Compute wait-time metrics for one station."""
        mu = service_rate or self.default_service_rate
        state = QueueState(
            station_id=station_id,
            station_name=station_name,
            num_ports=max(1, num_ports),
            arrival_rate=max(0.01, arrival_rate),
            service_rate=max(0.01, mu),
            current_queue=current_queue,
        )
        return state

    # ─── Arrival Rate Estimation ──────────────────────────────────────────────

    def estimate_arrival_rate(
        self,
        sessions_df: pd.DataFrame,
        station_id: int,
        hour_of_day: Optional[int] = None,
        day_of_week: Optional[int] = None,
    ) -> float:
        """
        Estimate λ from historical session data.
        Filters by hour / day if provided.
        """
        if sessions_df.empty or "station_id" not in sessions_df.columns:
            return self.default_arrival_rate

        mask = sessions_df["station_id"] == station_id
        if hour_of_day is not None and "hour_of_day" in sessions_df.columns:
            mask &= sessions_df["hour_of_day"] == hour_of_day
        if day_of_week is not None and "day_of_week" in sessions_df.columns:
            mask &= sessions_df["day_of_week"] == day_of_week

        subset = sessions_df[mask]
        if len(subset) < 10:
            return self.default_arrival_rate

        # Sessions per hour = count / total_hours_observed
        if "session_start" in subset.columns:
            time_range_hr = (
                pd.to_datetime(subset["session_start"]).max()
                - pd.to_datetime(subset["session_start"]).min()
            ).total_seconds() / 3600
            time_range_hr = max(1, time_range_hr)
        else:
            time_range_hr = len(subset)

        lambda_est = len(subset) / time_range_hr
        return round(max(0.1, lambda_est), 3)

    def estimate_service_rate(
        self,
        sessions_df: pd.DataFrame,
        station_id: Optional[int] = None,
    ) -> float:
        """Estimate μ = 1 / avg_session_duration_hr."""
        mask = pd.Series(True, index=sessions_df.index)
        if station_id and "station_id" in sessions_df.columns:
            mask = sessions_df["station_id"] == station_id

        dur_col = "session_duration_min"
        if dur_col in sessions_df.columns:
            avg_min = sessions_df[mask][dur_col].median()
            if avg_min and avg_min > 0:
                return round(60 / avg_min, 4)
        return self.default_service_rate

    # ─── Network Analysis ─────────────────────────────────────────────────────

    def analyze_network(
        self,
        stations_df: pd.DataFrame,
        sessions_df: Optional[pd.DataFrame] = None,
        hour_of_day: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute queue metrics for all stations."""
        results = []
        for _, station in stations_df.iterrows():
            sid = station.get("station_id", 0)
            ports = int(station.get("num_ports", 2))

            if sessions_df is not None:
                lam = self.estimate_arrival_rate(sessions_df, sid, hour_of_day)
                mu = self.estimate_service_rate(sessions_df, sid)
            else:
                lam = self.default_arrival_rate
                mu = self.default_service_rate

            state = self.compute_wait(
                num_ports=ports,
                arrival_rate=lam,
                service_rate=mu,
                station_id=sid,
                station_name=str(station.get("name", f"Station {sid}")),
            )
            results.append(state.to_dict())

        df = pd.DataFrame(results).sort_values("avg_wait_min")
        return df

    # ─── Capacity Planning ────────────────────────────────────────────────────

    def recommend_capacity(
        self,
        arrival_rate: float,
        service_rate: float,
        target_wait_min: float = 15.0,
        max_ports: int = 20,
    ) -> Dict:
        """Find minimum ports c to achieve target_wait_min."""
        for c in range(1, max_ports + 1):
            state = self.compute_wait(
                num_ports=c,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
            )
            if state.system_stable and state.avg_wait_min <= target_wait_min:
                return {
                    "recommended_ports": c,
                    "predicted_wait_min": state.avg_wait_min,
                    "utilization_pct": state.rho * 100,
                    "erlang_c": state.erlang_c,
                }
        return {
            "recommended_ports": max_ports,
            "predicted_wait_min": float("inf"),
            "utilization_pct": 100.0,
            "note": f"Cannot achieve {target_wait_min} min wait with ≤{max_ports} ports",
        }

    # ─── Wait-Time Sensitivity ────────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        num_ports: int,
        service_rate: float,
        arrival_rates: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """How does wait time change as λ increases?"""
        if arrival_rates is None:
            max_stable = num_ports * service_rate * 0.99
            arrival_rates = np.linspace(0.5, max_stable, 20).tolist()

        rows = []
        for lam in arrival_rates:
            state = self.compute_wait(num_ports, lam, service_rate)
            rows.append({
                "arrival_rate": round(lam, 3),
                "rho": state.rho,
                "avg_wait_min": state.avg_wait_min if state.system_stable else None,
                "avg_queue_length": state.avg_queue_length if state.system_stable else None,
                "system_stable": state.system_stable,
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    model = MMcQueueModel()

    # Single station demo
    state = model.compute_wait(
        num_ports=4,
        arrival_rate=8.0,
        service_rate=3.5,
        current_queue=3,
        station_id=1,
        station_name="Downtown Hub",
    )
    print("=== Single Station ===")
    for k, v in state.to_dict().items():
        print(f"  {k:30s}: {v}")

    # Capacity planning
    print("\n=== Capacity Planning ===")
    rec = model.recommend_capacity(arrival_rate=12.0, service_rate=3.0, target_wait_min=10)
    print(rec)

    # Sensitivity
    print("\n=== Sensitivity Analysis ===")
    sensitivity = model.sensitivity_analysis(num_ports=6, service_rate=3.0)
    print(sensitivity.to_string(index=False))
