"""Pydantic-validated configuration for AIT.

All configuration is loaded from config.yaml with environment variable overrides.
Validation catches misconfigurations BEFORE the bot starts trading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class TradingConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    universe: list[str] = ["SPY", "QQQ", "AAPL"]
    scan_interval_seconds: int = Field(default=300, ge=30, le=3600)
    max_daily_trades: int = Field(default=5, ge=1, le=50)
    trading_hours_only: bool = True


class AccountConfig(BaseModel):
    pdt_protection: bool = True
    pdt_account_under_25k: bool = True


class PositionConfig(BaseModel):
    max_open_positions: int = Field(default=5, ge=1, le=20)
    max_position_pct: float = Field(default=0.05, ge=0.01, le=0.25)
    max_portfolio_delta: float = Field(default=0.30, ge=0.05, le=1.0)
    max_portfolio_risk_pct: float = Field(default=0.02, ge=0.005, le=0.10)


class RiskConfig(BaseModel):
    max_daily_loss_pct: float = Field(default=0.02, ge=0.005, le=0.10)
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    pause_minutes_after_losses: int = Field(default=30, ge=5, le=120)
    max_api_failures: int = Field(default=5, ge=2, le=20)
    min_confidence: float = Field(default=0.65, ge=0.50, le=0.95)


class OptionsConfig(BaseModel):
    delta_range: list[float] = [0.20, 0.50]
    dte_range: list[int] = [14, 45]
    min_open_interest: int = Field(default=100, ge=10)
    min_volume: int = Field(default=50, ge=10)
    max_bid_ask_spread_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    strategies: list[str] = [
        "long_call",
        "long_put",
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
    ]

    @field_validator("delta_range")
    @classmethod
    def validate_delta_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or not (0 < v[0] < v[1] < 1):
            raise ValueError("delta_range must be [low, high] where 0 < low < high < 1")
        return v

    @field_validator("dte_range")
    @classmethod
    def validate_dte_range(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or not (1 <= v[0] < v[1] <= 365):
            raise ValueError("dte_range must be [min_dte, max_dte] where 1 <= min < max <= 365")
        return v


class BacktestConfig(BaseModel):
    initial_capital: float = Field(default=100_000.0, ge=1_000.0, le=10_000_000.0)
    position_size_pct: float = Field(default=0.05, ge=0.01, le=0.50,
        description="Fraction of capital risked per trade (max-loss basis).")
    wing_floor_dollars: float = Field(default=5.0, ge=0.50, le=50.0,
        description="Hard minimum spread width in dollars (safety floor only). "
                    "Wing sizing is now primarily driven by wing_k × expected_move.")
    wing_k: float = Field(default=1.0, ge=0.1, le=3.0,
        description="Vol-scaled wing multiplier: wing_width = wing_k × price × IV × sqrt(DTE/365). "
                    "Optuna optimizes this per walk-forward window. "
                    "1.0 = 1-sigma expected move. wing_floor_dollars is the hard minimum.")
    iv_floor: float = Field(default=0.20, ge=0.05, le=1.0,
        description="Minimum synthetic IV used for option pricing. "
                    "Prevents near-zero credits in calm markets.")
    delta_iv_scale: float = Field(default=0.0, ge=0.0, le=1.0,
        description="IV-driven delta scaling for strangles. "
                    "0=static delta, 1=full IV response. "
                    "High IV → lower effective delta → further OTM strikes.")
    max_concurrent_positions: int = Field(default=1, ge=1, le=5,
        description="Maximum number of simultaneously open positions. "
                    "1 = original single-position behavior. "
                    "3 = up to 3 concurrent iron condors / strangles.")
    max_entry_vol_annual: float = Field(default=0.80, ge=0.15, le=1.50,
        description="Maximum 10-day realized vol (annualized) allowed for iron condor / "
                    "short strangle entry. Entries above this are skipped. "
                    "Optuna tunes per window in range [0.25, 0.90].")
    optimize_n_trials: int = Field(default=50, ge=5, le=500,
        description="Optuna trials per walk-forward window.")
    optimize_patience: int = Field(default=20, ge=0, le=500,
        description="Early stopping: halt after this many consecutive non-improving trials. "
                    "0 = disabled.")
    optimize_min_trades: int = Field(default=10, ge=1, le=100,
        description="Min trade count for full objective score. "
                    "Trials below this are penalised quadratically; < 3 trades always scores −100.")
    # Intraday execution params (Fix 1 / Gap H)
    scan_interval_minutes: int = Field(default=60, ge=5, le=240,
        description="How often (minutes) to scan for entry signals within a trading session.")
    entry_window_start_et: str = Field(default="10:30",
        description="Earliest allowed entry time (ET). Signals before this are ignored.")
    entry_window_end_et: str = Field(default="15:30",
        description="Latest allowed entry time (ET). No new entries after this time.")
    limit_order_timeout_bars: int = Field(default=3, ge=1, le=20,
        description="Cancel a pending limit order after this many 5-min bars without a fill.")
    # Options spread model params (Fix 5)
    spread_base: float = Field(default=0.03, ge=0.005, le=0.20,
        description="Base per-leg half-spread cost ($). Represents minimum friction in liquid markets.")
    spread_iv_sensitivity: float = Field(default=0.10, ge=0.0, le=0.50,
        description="Additional half-spread per unit of IV above 0.20. Higher IV → wider market.")
    spread_dte_sensitivity: float = Field(default=0.005, ge=0.0, le=0.05,
        description="Additional half-spread per DTE below 21. Near-expiry options are wider.")
    spread_cap: float = Field(default=0.15, ge=0.01, le=0.50,
        description="Maximum per-leg half-spread ($). Prevents unrealistic spread in stress regimes.")
    # Fractal regime params (Gap Z5) — also used by live orchestrator for parity with backtest
    hurst_regime_threshold: float = Field(default=0.20, ge=0.05, le=0.50,
        description="Hurst scale-spread above which fractal confidence penalty is applied.")
    hurst_regime_penalty: float = Field(default=0.10, ge=0.0, le=0.30,
        description="Confidence deducted when fractal regime is chaotic.")
    multifractal_max_width: float = Field(default=0.50, ge=0.20, le=0.80,
        description="Multifractal width above which fractal confidence penalty is applied.")


class MLConfig(BaseModel):
    ensemble_weights: dict[str, float] = {"xgboost": 0.5, "lightgbm": 0.5}
    retrain_interval_days: int = Field(default=7, ge=1, le=30)
    lookback_days: int = Field(default=504, ge=60, le=2520)
    min_training_samples: int = Field(default=100, ge=30)

    @field_validator("ensemble_weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        return v


class MetaLabelConfig(BaseModel):
    enabled: bool = True
    min_probability: float = Field(default=0.50, ge=0.30, le=0.80)
    retrain_with_primary: bool = True  # Retrain when primary model retrains


class SentimentSourcesConfig(BaseModel):
    news: bool = True
    fear_greed: bool = True
    finbert: bool = True


class SentimentConfig(BaseModel):
    enabled: bool = True
    weight: float = Field(default=0.20, ge=0.0, le=0.50)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    sources: SentimentSourcesConfig = SentimentSourcesConfig()
    ib_news_weight: float = Field(default=0.20, ge=0.0, le=0.50)


class ExitConfig(BaseModel):
    trailing_stop_pct: float = Field(default=0.25, ge=0.10, le=0.50)
    breakeven_trigger_pct: float = Field(default=0.30, ge=0.10, le=0.80)
    partial_exit_levels: list[dict] = [
        {"pnl_pct": 0.50, "close_pct": 0.33},
        {"pnl_pct": 1.00, "close_pct": 0.33},
    ]
    time_decay_scaling: bool = True
    volatility_adjusted_stops: bool = True
    initial_stop_loss_pct: float = Field(default=0.50, ge=0.15, le=0.75)
    auto_hedge: bool = False


class LearningConfig(BaseModel):
    enabled: bool = True
    lookback_days: int = Field(default=30, ge=7, le=180)
    max_adaptations_per_cycle: int = Field(default=3, ge=1, le=10)
    min_insight_confidence: float = Field(default=0.60, ge=0.40, le=0.95)
    min_confidence_floor: float = Field(default=0.50, ge=0.40, le=0.80)
    min_confidence_ceiling: float = Field(default=0.90, ge=0.70, le=0.99)
    paper_trading_mode: bool = Field(
        default=False,
        description=(
            "When True, disables all live-only overlays (adaptor confidence/stop/trailing/"
            "take-profit overrides, Thompson sampling reranking, meta-labeler gate, options flow gate) "
            "to produce a clean backtest-equivalent paper-trading run for P&L comparison."
        ),
    )


class TelegramConfig(BaseModel):
    enabled: bool = True
    send_trades: bool = True
    send_errors: bool = True
    send_daily_summary: bool = True
    send_circuit_breaker: bool = True


class NotificationsConfig(BaseModel):
    telegram: TelegramConfig = TelegramConfig()


class DashboardConfig(BaseModel):
    enabled: bool = True
    port: int = Field(default=8501, ge=1024, le=65535)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "logs/ait.log"
    max_bytes: int = Field(default=10_485_760, ge=1_048_576)
    backup_count: int = Field(default=5, ge=1, le=20)


class IBKREnvConfig(BaseSettings):
    """IBKR connection settings from environment variables."""

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_account: str = ""

    model_config = {"env_file": ".env", "env_prefix": "", "case_sensitive": False, "extra": "ignore"}


class APIKeysConfig(BaseSettings):
    """API keys from environment variables — never stored in config.yaml."""

    polygon_api_key: str = ""
    finnhub_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    model_config = {"env_file": ".env", "env_prefix": "", "case_sensitive": False, "extra": "ignore"}


class Settings(BaseModel):
    """Root configuration — validated on startup."""

    trading: TradingConfig = TradingConfig()
    account: AccountConfig = AccountConfig()
    positions: PositionConfig = PositionConfig()
    risk: RiskConfig = RiskConfig()
    options: OptionsConfig = OptionsConfig()
    backtest: BacktestConfig = BacktestConfig()
    ml: MLConfig = MLConfig()
    meta_label: MetaLabelConfig = MetaLabelConfig()
    exit: ExitConfig = ExitConfig()
    learning: LearningConfig = LearningConfig()
    sentiment: SentimentConfig = SentimentConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()

    # Loaded from environment
    ibkr: IBKREnvConfig = IBKREnvConfig()
    api_keys: APIKeysConfig = APIKeysConfig()


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Load settings from YAML file, with env var overrides for secrets."""
    config_path = Path(config_path)

    yaml_data = {}
    if config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    # IBKR and API keys come from environment only
    yaml_data["ibkr"] = IBKREnvConfig().model_dump()
    yaml_data["api_keys"] = APIKeysConfig().model_dump()

    return Settings.model_validate(yaml_data)
