

from __future__ import annotations
import math
from typing import Optional, Any, Dict, Tuple

def calculate_tau_prime(delta_sigma: float,
                        cfg: Optional[Any] = None) -> float:
    """
    Emergent time mapping:
      tau_prime = tau_min + (tau_max - tau_min) / (1 + exp(tau_k * (delta_sigma - tau_theta)))
    As |Δσ| increases, tau_prime decreases (time contraction).
    """
    tau_min   = getattr(cfg, "TAU_PRIME_MIN", 0.1) if cfg is not None else 0.1
    tau_max   = getattr(cfg, "TAU_PRIME_MAX", 1.0) if cfg is not None else 1.0
    tau_k     = getattr(cfg, "TAU_PRIME_K", 10.0) if cfg is not None else 10.0
    tau_theta = getattr(cfg, "TAU_PRIME_THETA", 0.2) if cfg is not None else 0.2

    tau_min = float(tau_min)
    tau_max = float(tau_max)
    tau_k = float(tau_k)
    tau_theta = float(tau_theta)

    ds = max(0.0, float(delta_sigma))
    denom = 1.0 + math.exp(tau_k * (ds - tau_theta))
    return float(tau_min + (tau_max - tau_min) / denom)


def compute_delta_sigma_emergent(
    sigma_t: float,
    sigma_prev: Optional[float],
    S_t: Optional[float],
    S_prev: Optional[float],
    phasic_active: Optional[int],
    cfg: Optional[Any],
    state: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Build an emergent |Δσ| with an autopoietic floor so time never freezes:
      Δσ_raw = |σ_t - σ_{t-1}|
      ΔS = |S_t - S_{t-1}| (optional auxiliary drive)
      ema = β*ema + (1-β)*Δσ_raw
      Δσ_eff = α*ema + (1-α)*ΔS + ε + boost(phasic)

    Returns (delta_sigma_eff, updated_state).
    """
    beta = float(getattr(cfg, "TAU_PRIME_EMA_BETA", 0.9)) if cfg is not None else 0.9
    alpha = float(getattr(cfg, "TAU_PRIME_BLEND_SIGMA", 0.85)) if cfg is not None else 0.85
    eps = float(getattr(cfg, "TAU_PRIME_EPS", 1e-4)) if cfg is not None else 1e-4
    phasic_boost = float(getattr(cfg, "TAU_PRIME_PHASIC_BOOST", 0.05)) if cfg is not None else 0.05

    d_sigma = 0.0 if sigma_prev is None else abs(float(sigma_t) - float(sigma_prev))
    d_S = 0.0
    if S_t is not None and S_prev is not None:
        d_S = abs(float(S_t) - float(S_prev))

    ema = state.get("ema_dsigma", 0.0)
    ema = beta * ema + (1.0 - beta) * d_sigma
    state["ema_dsigma"] = ema

    base = alpha * ema + (1.0 - alpha) * d_S
    if phasic_active is not None and int(phasic_active) == 1:
        base += phasic_boost

    delta_sigma_eff = max(base + eps, eps)
    return float(delta_sigma_eff), state
