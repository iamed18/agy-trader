from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def ensure_utc(dt: datetime) -> datetime:
    """Validate that a datetime is timezone-aware and convert it to UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return dt.astimezone(UTC)


NY_TZ = ZoneInfo("America/New_York")


def compute_headline_hash(headline: str, published_at: datetime) -> str:
    """Compute SHA-256 hash of normalized headline + America/New_York date."""
    stripped = headline.strip()
    if not stripped:
        raise ValueError("headline must not be empty")
    utc_dt = ensure_utc(published_at)
    ny_date = utc_dt.astimezone(NY_TZ).date().isoformat()
    lowered = stripped.lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    if not normalized:
        normalized = lowered
    payload = f"{normalized}|{ny_date}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


UtcDatetime = Annotated[datetime, AfterValidator(ensure_utc)]


def _clean_symbol(v: str) -> str:
    cleaned = v.strip().upper()
    if not cleaned:
        raise ValueError("Symbol must not be empty")
    return cleaned


class AssetType(StrEnum):
    etf = "etf"
    equity = "equity"


class OrderSide(StrEnum):
    buy = "buy"
    sell = "sell"


class PositionSide(StrEnum):
    long = "long"
    short = "short"


class _StrictModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)


class UniverseSymbol(_StrictModel):
    symbol: str
    asset_type: AssetType
    sector: str
    shortable_default: bool = True

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _clean_symbol(v)


class Bar(_StrictModel):
    symbol: str
    timestamp: UtcDatetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    vwap: float | None = Field(default=None, gt=0.0)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _clean_symbol(v)

    @model_validator(mode="after")
    def _validate_ohlc(self) -> Bar:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(
                f"high ({self.high}) must be >= open ({self.open}) and close ({self.close})"
            )
        if self.low > self.open or self.low > self.close:
            raise ValueError(
                f"low ({self.low}) must be <= open ({self.open}) and close ({self.close})"
            )
        return self


class NewsArticle(_StrictModel):
    article_id: str
    published_at: UtcDatetime
    headline: str
    summary: str = ""
    source: str = "benzinga"
    symbols: list[str]
    headline_hash: str = ""
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    scorer_model: str | None = None

    @field_validator("article_id")
    @classmethod
    def _validate_article_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("article_id must not be empty")
        return cleaned

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return [s.strip().upper() for s in v if isinstance(s, str) and s.strip()]

    @model_validator(mode="after")
    def _populate_headline_hash(self) -> NewsArticle:
        if not self.headline.strip():
            raise ValueError("headline must not be empty")
        if not self.headline_hash.strip():
            self.headline_hash = compute_headline_hash(self.headline, self.published_at)
        return self


class EarningsEvent(_StrictModel):
    symbol: str
    earnings_date: date

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _clean_symbol(v)


class MacroPoint(_StrictModel):
    as_of_date: date
    series_id: str
    value: float

    @field_validator("series_id")
    @classmethod
    def _validate_series_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("series_id must not be empty")
        return cleaned


class ControlDials(_StrictModel):
    pause_new_entries: bool = False
    exposure_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_shorting: bool = True
    blacklisted_symbols: dict[str, UtcDatetime] = Field(default_factory=dict)
    updated_by: str = "system"
    reason: str = ""
    updated_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("exposure_multiplier")
    @classmethod
    def _validate_finite_multiplier(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("exposure_multiplier must be finite")
        return v

    @field_validator("blacklisted_symbols", mode="before")
    @classmethod
    def _normalize_blacklist_keys(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        return {
            str(k).strip().upper(): val
            for k, val in v.items()
            if str(k).strip()
        }

    def active_blacklist(self, as_of: datetime) -> set[str]:
        as_of_utc = ensure_utc(as_of)
        return {
            symbol
            for symbol, expiry in self.blacklisted_symbols.items()
            if expiry > as_of_utc
        }


class LLMUsageRecord(_StrictModel):
    timestamp: UtcDatetime
    tier: int = Field(ge=0, le=2)
    model: str
    task: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    estimated_usd: float = Field(ge=0.0)


class AssetRiskMetadata(_StrictModel):
    symbol: str
    asset_type: AssetType
    sector: str
    shortable: bool = True
    easy_to_borrow: bool = True

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _clean_symbol(v)


class SignalRow(_StrictModel):
    as_of: UtcDatetime
    symbol: str
    trend_score: float
    pullback_score: float
    news_score: float
    news_velocity_z: float
    composite_score: float
    macro_regime_scale: float
    atr_pct: float = Field(gt=0.0)
    ann_vol: float = Field(ge=0.0)
    ret_5d: float

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _clean_symbol(v)


class ProposedOrder(_StrictModel):
    symbol: str
    side: OrderSide
    qty: float = Field(gt=0.0)
    target_side: PositionSide | None = None
    is_risk_reducing: bool = False
    reference_price: float = Field(gt=0.0)
    stop_loss_price: float | None = Field(default=None, gt=0.0)
    reason: str = ""

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        return _clean_symbol(v)


class RiskVerdict(_StrictModel):
    approved: bool
    order: ProposedOrder
    adjusted_qty: float = Field(default=0.0, ge=0.0)
    stop_loss_price: float | None = Field(default=None, gt=0.0)
    violations: list[str] = Field(default_factory=list)
