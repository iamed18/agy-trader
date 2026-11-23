from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agy_trader.core.models import UniverseSymbol

LIVE_CONFIRMATION_PHRASE = "I_ACCEPT_REAL_MONEY_RISK"


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    live_trading_confirmation: str = ""
    alpaca_data_feed: Literal["iex", "sip"] = "iex"
    fred_api_key: str = ""
    finnhub_api_key: str = ""
    sec_edgar_user_agent: str = ""
    discord_webhook_url: str = ""
    nlp_backend: Literal["lexicon", "finbert"] = "lexicon"
    blackout_period: str = ""
    agy_data_dir: Path = Path("data")
    agy_config_dir: Path = Path("config")


def validate_trading_mode(settings: EnvSettings) -> None:
    """Enforce a double-lock before allowing live trading."""
    if settings.alpaca_paper:
        return

    missing: list[str] = []
    if not settings.alpaca_api_key.strip():
        missing.append("alpaca_api_key")
    if not settings.alpaca_secret_key.strip():
        missing.append("alpaca_secret_key")

    if missing or settings.live_trading_confirmation != LIVE_CONFIRMATION_PHRASE:
        raise ValueError(
            "Live trading double-lock failed: non-paper mode requires non-empty "
            "alpaca_api_key, alpaca_secret_key, and live_trading_confirmation == "
            f"{LIVE_CONFIRMATION_PHRASE!r}."
        )


_TIME_HH_MM_RE = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d")


class StrategyConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)

    rebalance_checkpoints_et: list[str] = Field(
        default_factory=lambda: ["10:00", "13:00", "15:40"],
        min_length=1,
    )
    momentum_mode: Literal["legacy", "m2_residual"] = "legacy"
    m2_long_lookback: int = Field(default=252, ge=60, le=504)
    m2_long_skip: int = Field(default=21, ge=0, le=63)
    m2_med_lookback: int = Field(default=126, ge=40, le=252)
    m2_med_skip: int = Field(default=10, ge=0, le=42)
    m2_w_long: float = Field(default=0.50, ge=0.0, le=1.0)
    m2_w_med: float = Field(default=0.25, ge=0.0, le=1.0)
    m2_w_residual: float = Field(default=0.25, ge=0.0, le=1.0)
    m2_w_raw_long: float = Field(default=0.0, ge=0.0, le=1.0)
    m2_w_intermed: float = Field(default=0.0, ge=0.0, le=1.0)
    m2_w_52w_high: float = Field(default=0.0, ge=0.0, le=1.0)
    m2_w_reg_slope: float = Field(default=0.0, ge=0.0, le=1.0)
    m2_w_liquidity: float = Field(default=0.0, ge=0.0, le=1.0)
    reg_slope_lookback: int = Field(default=126, ge=40, le=252)
    w_trend: float = Field(default=0.45, ge=0.0, le=1.0)
    w_pullback: float = Field(default=0.25, ge=0.0, le=1.0)
    w_news_sentiment: float = Field(default=0.30, ge=0.0, le=1.0)
    w_insider: float = Field(default=0.0, ge=0.0, le=1.0)
    insider_lookback_days: int = Field(default=60, ge=5, le=180)
    model_news_sources: list[str] | None = Field(
        default_factory=lambda: ["benzinga", "fiscal.ai", "finnhub"]
    )
    news_half_life_hours: float = Field(default=24.0, gt=0.0)
    news_lookback_hours: float = Field(default=72.0, gt=0.0)
    top_k_long: int = Field(default=8, ge=1)
    top_k_short: int = Field(default=6, ge=0)
    min_signal_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    rebalance_deadband_pct: float = Field(default=0.015, ge=0.0, le=0.5)
    allow_fractional_longs: bool = True
    rebalance_every_n_days: int = Field(default=1, ge=1, le=21)
    rebalance_schedule: Literal["every_n_days", "monthly_first", "monthly_last"] = "every_n_days"
    hold_rank_buffer: int = Field(default=0, ge=0, le=50)
    exit_signal_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    abs_momentum_filter: bool = False
    use_rf_hurdle_for_abs_momentum: bool = True
    short_only_in_bear_regime: bool = False
    max_pair_corr_60d: float | None = Field(default=None, ge=-1.0, le=1.0)
    corr_penalty_eta: float = Field(default=0.0, ge=0.0, le=5.0)
    score_tilt_power: float = Field(default=0.0, ge=0.0, le=2.0)
    waterfill_weights: bool = False
    exempt_safe_havens_from_macro_scale: bool = False
    safe_haven_rotation_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    exclude_safe_havens_from_equity_slots: bool = False
    spy_sma200_defensive_scale: float | None = Field(default=None, ge=0.0, le=1.0)
    spy_ema50_dual_scale: float | None = Field(default=None, ge=0.0, le=1.0)
    spy_vol_surge_scale: float | None = Field(default=None, ge=0.0, le=1.0)
    spy_vol_surge_thresh: float = Field(default=1.65, gt=0.0)
    spy_ret10_crash_thresh: float = Field(default=-0.055, lt=0.0)
    spy_ret10_crash_scale: float | None = Field(default=None, ge=0.0, le=1.0)
    use_sector_residual: bool = False
    w_fip: float = Field(default=0.0, ge=0.0, le=1.0)
    w_jump_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    fip_gate_quantile: float | None = Field(default=None, ge=0.05, le=0.95)
    max_stock_ann_vol: float | None = Field(default=None, gt=0.0, le=5.0)
    trend_above_sma200_filter: bool = False
    vol_sizing_mode: Literal["inv_atr", "capped_inv_vol", "equal", "liquidity_pow"] = "inv_atr"
    liquidity_weight_power: float = Field(default=0.5, ge=0.0, le=1.0)
    vol_sizing_floor: float = Field(default=0.18, gt=0.0, le=1.0)
    vol_sizing_cap: float = Field(default=0.45, gt=0.0, le=2.0)
    portfolio_vol_target_ann: float | None = Field(default=None, gt=0.0, le=1.0)
    rebalance_tranches: int = Field(default=1, ge=1, le=8)
    regime_mode: Literal["legacy_stacked", "daniel_moskowitz", "none", "sma200_only"] = "legacy_stacked"
    w_pead: float = Field(default=0.0, ge=0.0, le=1.0)
    finalist_pool_size: int = Field(default=40, ge=5, le=200)

    @field_validator("model_news_sources")
    @classmethod
    def _validate_model_news_sources(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [s.strip().lower() for s in v if s.strip()]

    @field_validator("rebalance_checkpoints_et")
    @classmethod
    def _validate_rebalance_checkpoints_et(cls, v: list[str]) -> list[str]:
        for item in v:
            if _TIME_HH_MM_RE.fullmatch(item) is None:
                raise ValueError(
                    f"Invalid rebalance checkpoint {item!r}: expected strict 24-hour 'HH:MM'"
                )
        return v

    @model_validator(mode="after")
    def _validate_weights(self) -> StrategyConfig:
        total = (
            self.w_trend
            + self.w_pullback
            + self.w_news_sentiment
            + self.w_insider
            + self.w_fip
            + self.w_pead
        )
        if total <= 0.0:
            raise ValueError("Strategy signal weights must sum to > 0")
        if self.vol_sizing_cap < self.vol_sizing_floor:
            raise ValueError("vol_sizing_cap must be >= vol_sizing_floor")
        if self.momentum_mode == "m2_residual":
            m2_total = (
                self.m2_w_long
                + self.m2_w_med
                + self.m2_w_residual
                + self.m2_w_raw_long
                + self.m2_w_intermed
                + self.m2_w_52w_high
                + self.m2_w_reg_slope
                + self.m2_w_liquidity
            )
            if m2_total <= 0.0:
                raise ValueError(
                    "m2_residual weights (m2_w_long, m2_w_med, m2_w_residual, "
                    "m2_w_raw_long, m2_w_intermed, m2_w_52w_high, m2_w_reg_slope, "
                    "m2_w_liquidity) must sum to > 0"
                )
        else:
            if (
                self.w_fip > 0.0
                or self.w_jump_penalty > 0.0
                or self.use_sector_residual
                or self.m2_w_raw_long > 0.0
                or self.m2_w_intermed > 0.0
                or self.m2_w_52w_high > 0.0
                or self.m2_w_reg_slope > 0.0
                or self.m2_w_liquidity > 0.0
                or self.fip_gate_quantile is not None
                or self.max_stock_ann_vol is not None
                or self.trend_above_sma200_filter
                or self.vol_sizing_mode == "liquidity_pow"
            ):
                raise ValueError(
                    "V3/V4 residual, FIP, raw/intermed/52w/slope/liquidity momentum, "
                    "liquidity_pow sizing, and stage-1 gate fields require momentum_mode == 'm2_residual'"
                )
        if self.news_lookback_hours < self.news_half_life_hours:
            raise ValueError("news_lookback_hours must be >= news_half_life_hours")
        if (
            self.exit_signal_threshold is not None
            and self.exit_signal_threshold > self.min_signal_threshold
        ):
            raise ValueError("exit_signal_threshold must be <= min_signal_threshold")
        return self


class RiskConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)

    max_long_position_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    max_short_position_pct: float = Field(default=0.06, ge=0.0, le=1.0)
    max_gross_exposure: float = Field(default=1.00, gt=0.0, le=2.0)
    min_net_exposure: float = Field(default=-0.25, ge=-1.0, le=1.0)
    max_net_exposure: float = Field(default=0.60, ge=-1.0, le=1.0)
    max_sector_gross_pct: float = Field(default=0.25, gt=0.0, le=1.0)
    daily_loss_halt_pct: float = Field(default=0.02, gt=0.0, lt=1.0)
    max_drawdown_halt_pct: float = Field(default=0.08, gt=0.0, lt=1.0)
    max_drawdown_cooldown_days: int | None = Field(default=None, ge=1, le=252)
    delisting_haircut_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    earnings_blackout_days: int = Field(default=2, ge=0, le=30)
    restatement_blackout_days: int = Field(default=10, ge=0, le=60)
    stop_loss_atr_mult_long: float = Field(default=2.5, gt=0.0)
    stop_loss_atr_mult_short: float = Field(default=2.0, gt=0.0)
    short_squeeze_5d_return_cap: float = Field(default=0.20, gt=0.0)
    short_squeeze_ann_vol_cap: float = Field(default=0.90, gt=0.0)
    max_data_staleness_minutes: int = Field(default=15, ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> RiskConfig:
        if self.max_short_position_pct > self.max_long_position_pct:
            raise ValueError("max_short_position_pct must be <= max_long_position_pct")
        if self.min_net_exposure > self.max_net_exposure:
            raise ValueError("min_net_exposure must be <= max_net_exposure")
        if self.max_drawdown_halt_pct < self.daily_loss_halt_pct:
            raise ValueError("max_drawdown_halt_pct must be >= daily_loss_halt_pct")
        return self


class BacktestConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)

    initial_capital: float = Field(default=100000.0, gt=0.0)
    etf_slippage_bps: float = Field(default=5.0, ge=0.0)
    equity_slippage_bps: float = Field(default=10.0, ge=0.0)
    short_borrow_apr: float = Field(default=0.015, ge=0.0, le=1.0)
    sec_sell_fee_bps: float = Field(default=0.28, ge=0.0)
    risk_free_rate_apr: float = Field(default=0.045, ge=0.0, le=0.25)
    cash_yield_apr: float = Field(default=0.045, ge=0.0, le=0.25)
    use_fred_tbill_yield: bool = True
    execution_timing: Literal["same_close", "next_open"] = "same_close"
    benchmark_symbol: str = "SPY"


class LLMQuotaConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)

    daily_usd_cap: float = Field(default=1.50, ge=0.0)
    monthly_usd_cap: float = Field(default=25.00, ge=0.0)
    tier1_model: str = "flash"
    tier2_model: str = "pro"
    max_input_tokens_per_call: int = Field(default=8000, ge=1)

    @model_validator(mode="after")
    def _validate_caps(self) -> LLMQuotaConfig:
        if self.monthly_usd_cap < self.daily_usd_cap:
            raise ValueError("monthly_usd_cap must be >= daily_usd_cap")
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    llm_quota: LLMQuotaConfig = Field(default_factory=LLMQuotaConfig)


def load_app_config(config_dir: Path) -> AppConfig:
    """Load and validate strategy.yaml from config_dir."""
    strategy_path = config_dir / "strategy.yaml"
    if not strategy_path.exists():
        raise FileNotFoundError(f"Missing strategy config file: {strategy_path}")
    raw = yaml.safe_load(strategy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in {strategy_path}")
    return AppConfig.model_validate(raw)


def load_universe(config_dir: Path) -> list[UniverseSymbol]:
    """Load and validate universe.yaml from config_dir."""
    universe_path = config_dir / "universe.yaml"
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing universe config file: {universe_path}")
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "symbols" not in raw:
        raise ValueError(f"Expected mapping with 'symbols' key in {universe_path}")
    entries = raw["symbols"]
    if not isinstance(entries, list):
        raise TypeError(f"Expected 'symbols' list in {universe_path}")
    return [UniverseSymbol.model_validate(item) for item in entries]
