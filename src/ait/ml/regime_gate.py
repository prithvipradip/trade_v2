"""W67: the ONE fractal regime gate, shared by the live orchestrator and the research engine.

Why this exists. Live and research each carried their own copy of "what does a chaotic
fractal regime do to an entry", and the copies disagreed:

* live (``orchestrator.py``): a flat ``hurst_regime_penalty`` off the DIRECTIONAL confidence
  when ``hurst_scale_spread > threshold`` or ``multifractal_width > max`` -- inert for iron
  condors, which are gated on the range model's P(in range), not on direction;
* research (``engine.py``): a proportional penalty (equally inert for condors) plus a hard
  veto at ``max(threshold, 0.20) * hurst_hard_veto_multiplier``. The multiplier was
  honoured by the OOS Backtester (0 = off) but NOT passed to the optimizer's trial
  Backtester or the meta-label shadow Backtester, which therefore ran with the bare
  engine default 1.5 -- Optuna optimized under a gate the OOS evaluation never applied.

Both mechanisms are kept (operator decision, 2026-10-09): the veto as an explicit absolute
cutoff ``veto_spread`` (0 = off) and the penalty -- optionally ALSO acting on the range
model's probability (``penalty_on_range_prob``), the statistic condors are really gated
on. Defaults reproduce live behaviour today exactly. Pure function, no IO, so the
orchestrator, the engine, the gate study and the tests share one definition; a class guard
(tests/test_w67_regime_gate.py) fails if either caller re-implements the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

# Penalty shapes. "live_flat": one flat deduction when EITHER the spread or the
# multifractal width is over its limit (what the orchestrator has always done).
# "proportional": the research engine's historical shape -- the spread penalty scales with
# spread/threshold and the multifractal deduction is a second, additive flat one.
SHAPE_LIVE_FLAT = "live_flat"
SHAPE_PROPORTIONAL = "proportional"
SHAPES = (SHAPE_LIVE_FLAT, SHAPE_PROPORTIONAL)

# The research veto's historical floor: max(threshold, 0.20) * multiplier (Exp 20
# post-mortem: tighter thresholds blocked every entry on QQQ).
LEGACY_VETO_BASE_FLOOR = 0.20

# A very large spread must not drive the range-probability adjustment without bound.
RANGE_PROB_SCALE_CAP = 3.0


@dataclass(frozen=True)
class FractalGateConfig:
    threshold: float                 # hurst_regime_threshold (<= 0 disables the spread branch)
    penalty: float                   # hurst_regime_penalty
    multifractal_max_width: float
    veto_spread: float = 0.0         # absolute hard-veto cutoff; 0 = veto off
    penalty_on_range_prob: float = 0.0   # 0 = the range probability is never adjusted
    shape: str = SHAPE_LIVE_FLAT

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown penalty shape {self.shape!r}; expected one of {SHAPES}")
        if self.veto_spread < 0 or self.penalty_on_range_prob < 0:
            raise ValueError("veto_spread and penalty_on_range_prob must be >= 0")


@dataclass(frozen=True)
class FractalVerdict:
    veto: bool                 # hard veto fires (caller decides whether the structure is subject)
    directional_penalty: float  # deduct from DIRECTIONAL confidence (>= 0)
    range_prob_delta: float    # add to P(in range) (<= 0)
    over_spread: bool
    over_multifractal: bool

    @property
    def triggered(self) -> bool:
        return self.over_spread or self.over_multifractal


def resolve_veto_spread(threshold: float, multiplier: float | None,
                        explicit: float | None) -> float:
    """The absolute veto cutoff for one caller, once.

    ``explicit`` (``fractal_veto_spread``) wins when given. Otherwise the legacy research
    multiplier is translated with its original formula, so every existing caller that
    passes ``hurst_hard_veto_multiplier`` keeps its exact behaviour. 0 = veto off.
    """
    if explicit is not None:
        return max(float(explicit), 0.0)
    if multiplier is None or multiplier <= 0:
        return 0.0
    return max(float(threshold), LEGACY_VETO_BASE_FLOOR) * float(multiplier)


def fractal_gate(hurst_scale_spread: float, multifractal_width: float,
                 cfg: FractalGateConfig) -> FractalVerdict:
    """Evaluate the fractal regime for one entry decision."""
    spread = float(hurst_scale_spread)
    mf_w = float(multifractal_width)
    over_spread = cfg.threshold > 0 and spread > cfg.threshold
    over_mf = mf_w > 0 and mf_w > cfg.multifractal_max_width

    veto = bool(over_spread and cfg.veto_spread > 0 and spread > cfg.veto_spread)

    if cfg.shape == SHAPE_LIVE_FLAT:
        penalty = cfg.penalty if (over_spread or over_mf) else 0.0
    else:  # proportional (engine): spread penalty scales; multifractal is a second flat one
        penalty = 0.0
        if over_spread:
            penalty += cfg.penalty * (spread / cfg.threshold)
        if over_mf:
            penalty += cfg.penalty

    delta = 0.0
    if cfg.penalty_on_range_prob > 0 and (over_spread or over_mf):
        scale = spread / cfg.threshold if over_spread else 1.0
        delta = -cfg.penalty_on_range_prob * min(scale, RANGE_PROB_SCALE_CAP)

    return FractalVerdict(veto=veto, directional_penalty=penalty, range_prob_delta=delta,
                          over_spread=over_spread, over_multifractal=over_mf)


def apply_range_prob_delta(probability: float, verdict: FractalVerdict) -> float:
    """P(in range) after the fractal adjustment, kept a probability."""
    return min(1.0, max(0.0, float(probability) + verdict.range_prob_delta))
