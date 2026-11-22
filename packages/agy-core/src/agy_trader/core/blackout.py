from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agy_trader.core.config import load_universe
from agy_trader.core.models import AssetType, UniverseSymbol, ensure_utc

if TYPE_CHECKING:
    from agy_trader.storage.db import StateDB

ISSUER_SHARE_CLASS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"GOOG", "GOOGL"}),
    frozenset({"BRK.A", "BRK.B", "BRK-A", "BRK-B"}),
    frozenset({"FOXA", "FOX"}),
    frozenset({"BF.A", "BF.B", "BF-A", "BF-B"}),
    frozenset({"NWSA", "NWS"}),
    frozenset({"UHAL", "UHAL.B", "UHAL-B"}),
    frozenset({"LEN", "LEN.B", "LEN-B"}),
    frozenset({"MKC", "MKC.V", "MKC-V"}),
)
NON_EQUITY_ETF_SECTORS: frozenset[str] = frozenset({"Fixed Income", "Commodities"})
DEFAULT_ETF_CONCENTRATION_CAP: float = 0.235
DEFAULT_MAX_HOLDINGS_STALENESS_DAYS: int = 7

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_BLACKOUT_LINE_PREFIX_RE = re.compile(r"(?i)^(?:export\s+)?blackout")
_SYMBOL_TO_GROUP: dict[str, frozenset[str]] = {
    sym: grp for grp in ISSUER_SHARE_CLASS_GROUPS for sym in grp
}


def _canonical_share_class(symbol: str) -> str:
    return symbol.strip().upper().replace("-", ".")


class BlackoutParseError(ValueError):
    """Raised when BLACKOUT_PERIOD cannot be parsed or validated fail-safe."""


class EtfConcentrationInfo(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    etf_symbol: str
    sector: str
    observed_weight: float = 0.0
    conservative_weight: float = 0.0
    holdings_count: int = 0
    min_top_weight: float = 0.0
    stale_or_missing: bool = False
    frozen: bool = False
    reason: str = ""
    updated_at: str | None = None


class BlackoutResolution(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    raw_blackout_period: str = ""
    raw_symbols: tuple[str, ...] = ()
    direct_frozen_symbols: frozenset[str] = frozenset()
    etf_frozen_symbols: frozenset[str] = frozenset()
    frozen_symbols: frozenset[str] = frozenset()
    halted: bool = False
    halt_reason: str = ""
    has_warnings: bool = False
    warnings: tuple[str, ...] = ()
    etf_details: dict[str, EtfConcentrationInfo] = Field(default_factory=dict)
    etf_weights_summary: list[str] = Field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.halted or bool(self.frozen_symbols)

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "raw_blackout_period": self.raw_blackout_period,
            "raw_symbols": list(self.raw_symbols),
            "direct_frozen_symbols": sorted(self.direct_frozen_symbols),
            "etf_frozen_symbols": sorted(self.etf_frozen_symbols),
            "frozen_symbols": sorted(self.frozen_symbols),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "has_warnings": self.has_warnings,
            "warnings": list(self.warnings),
            "etf_weights_summary": list(self.etf_weights_summary),
        }


def issuer_share_class_group(symbol: str) -> frozenset[str]:
    """Return the share-class equivalence group for a symbol (or singleton)."""
    clean = symbol.strip().upper()
    return _SYMBOL_TO_GROUP.get(clean, frozenset({clean}))


def expand_share_class_symbols(symbols: Iterable[str]) -> frozenset[str]:
    """Expand ticker symbols across known multi-class issuer groups."""
    expanded: set[str] = set()
    for sym in symbols:
        clean = sym.strip().upper()
        if not clean:
            continue
        expanded.update(issuer_share_class_group(clean))
    return frozenset(expanded)


def _parse_env_value_literal(raw_val: str) -> str:
    """Strictly parse a single .env assignment RHS, rejecting unclosed/trailing quotes."""
    val = raw_val.strip()
    if not val:
        return ""
    if val[0] in ('"', "'"):
        quote = val[0]
        idx = 1
        closed = False
        chars: list[str] = []
        while idx < len(val):
            ch = val[idx]
            if quote == '"' and ch == "\\" and idx + 1 < len(val):
                nxt = val[idx + 1]
                if nxt in ('"', "\\"):
                    chars.append(nxt)
                    idx += 2
                    continue
                raise BlackoutParseError(
                    f"Unsupported escape sequence \\{nxt} in BLACKOUT_PERIOD"
                )
            if ch == quote:
                closed = True
                idx += 1
                break
            chars.append(ch)
            idx += 1
        if not closed:
            raise BlackoutParseError("Unclosed quote in BLACKOUT_PERIOD")
        trailing = val[idx:].strip()
        if trailing and not trailing.startswith("#"):
            raise BlackoutParseError(
                f"Unexpected trailing characters after quoted BLACKOUT_PERIOD: {trailing!r}"
            )
        return "".join(chars)

    # Unquoted value: allow empty inline comment ("# ...") or whitespace-separated inline comment
    if val.startswith("#"):
        return ""
    unquoted = re.split(r"\s+#", val, maxsplit=1)[0].strip()
    if '"' in unquoted or "'" in unquoted:
        raise BlackoutParseError("Unexpected quote character in unquoted BLACKOUT_PERIOD")
    return unquoted


def read_raw_blackout_from_env_file(env_file: Path) -> str:
    """Read and syntax-check BLACKOUT_PERIOD directly from a .env file on disk."""
    try:
        content = Path(env_file).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise BlackoutParseError(f"Failed to read {env_file}: {exc}") from exc

    found_value = ""
    seen_blackout = False
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _BLACKOUT_LINE_PREFIX_RE.match(stripped) is None:
            continue

        after_export = re.sub(r"^export[ \t]+", "", stripped)
        if "=" not in after_export:
            raise BlackoutParseError(
                f"Malformed BLACKOUT_PERIOD syntax on line {line_no} of {env_file}"
            )
        key_raw, val_raw = after_export.split("=", 1)
        if key_raw.strip() != "BLACKOUT_PERIOD":
            raise BlackoutParseError(
                f"Invalid blackout variable name {key_raw.strip()!r} on line {line_no} of {env_file}"
            )
        if seen_blackout:
            raise BlackoutParseError(
                f"Duplicate BLACKOUT_PERIOD entries in {env_file} (line {line_no})"
            )
        seen_blackout = True
        found_value = _parse_env_value_literal(val_raw)
    return found_value


def parse_blackout_period(raw: str) -> tuple[str, ...]:
    """Validate and normalize BLACKOUT_PERIOD into deduplicated uppercase ticker tokens."""
    text = raw.strip()
    if text and text[0] in ('"', "'"):
        quote = text[0]
        if len(text) < 2 or text[-1] != quote:
            raise BlackoutParseError("Unclosed quote in BLACKOUT_PERIOD")
        text = text[1:-1].strip()
    if '"' in text or "'" in text:
        raise BlackoutParseError("Unexpected quote character in BLACKOUT_PERIOD")
    if not text:
        return ()

    tokens = [tok for tok in re.split(r"[,\s]+", text) if tok]
    if not tokens:
        raise BlackoutParseError(f"Invalid empty token list in BLACKOUT_PERIOD: {raw!r}")

    normalized: list[str] = []
    for tok in tokens:
        upper = tok.upper()
        if _TICKER_RE.fullmatch(upper) is None:
            raise BlackoutParseError(
                f"Invalid ticker token {tok!r} in BLACKOUT_PERIOD (expected ^[A-Z][A-Z0-9.-]{{0,9}}$)"
            )
        if upper not in normalized:
            normalized.append(upper)
    return tuple(normalized)


def _halted_resolution(raw: str, reason: str) -> BlackoutResolution:
    return BlackoutResolution(
        raw_blackout_period=raw,
        halted=True,
        halt_reason=reason,
        has_warnings=True,
        warnings=(reason,),
    )


def load_universe_for_blackout(config_dir: Path) -> tuple[list[UniverseSymbol], str | None]:
    """Load universe.yaml without swallowing errors so non-empty blackouts fail closed."""
    try:
        return load_universe(Path(config_dir)), None
    except Exception as exc:  # noqa: BLE001 - surface load/validation error to fail closed
        return [], str(exc)


def resolve_live_blackout(
    *,
    env_file: Path | None = None,
    require_env_file: bool = False,
    fallback_raw: str = "",
    db: StateDB | None = None,
    universe: Sequence[UniverseSymbol] = (),
    universe_error: str | None = None,
    observed_broker_symbols: Iterable[str] = (),
    now_utc: datetime | None = None,
    etf_concentration_cap: float = DEFAULT_ETF_CONCENTRATION_CAP,
    max_holdings_staleness_days: int = DEFAULT_MAX_HOLDINGS_STALENESS_DAYS,
) -> BlackoutResolution:
    """Resolve current BLACKOUT_PERIOD freeze set and fail-closed halt state."""
    now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)

    raw = fallback_raw
    if env_file is not None:
        env_path = Path(env_file)
        if not env_path.exists():
            if require_env_file:
                return _halted_resolution("", f"Missing required .env file: {env_path}")
        else:
            try:
                raw = read_raw_blackout_from_env_file(env_path)
            except BlackoutParseError as exc:
                return _halted_resolution("", str(exc))

    try:
        raw_symbols = parse_blackout_period(raw)
    except BlackoutParseError as exc:
        return _halted_resolution(raw, str(exc))

    if not raw_symbols:
        return BlackoutResolution(raw_blackout_period=raw, raw_symbols=(), halted=False)

    if universe_error is not None:
        return _halted_resolution(
            raw,
            f"Failed to load universe metadata for ETF blackout verification: {universe_error}",
        )
    if not any(u.asset_type == AssetType.etf for u in universe):
        return _halted_resolution(
            raw,
            "Failed to load universe metadata for ETF blackout verification: no ETF symbols in universe",
        )

    direct_frozen = expand_share_class_symbols(raw_symbols)

    # Deduplicated issuer share-class groups corresponding to raw_symbols
    restricted_groups: list[frozenset[str]] = []
    seen_groups: set[frozenset[str]] = set()
    for sym in raw_symbols:
        grp = issuer_share_class_group(sym)
        if grp not in seen_groups:
            seen_groups.add(grp)
            restricted_groups.append(grp)

    etf_frozen: set[str] = set()
    etf_details: dict[str, EtfConcentrationInfo] = {}
    etf_weights_summary: list[str] = []
    warnings: list[str] = []
    has_warnings = False

    for entry in universe:
        if entry.asset_type != AssetType.etf:
            continue

        etf_sym = entry.symbol
        is_non_equity = entry.sector in NON_EQUITY_ETF_SECTORS
        meta = db.get_etf_holdings_meta(etf_sym) if db is not None else None
        holdings = db.get_etf_holdings(etf_sym) if db is not None else {}

        if not is_non_equity:
            stale_or_missing = False
            if meta is None:
                stale_or_missing = True
            else:
                h_count = int(meta["holdings_count"])
                total_weight = sum(holdings.values()) if holdings else 0.0
                if (
                    (h_count == 0 and not bool(meta["is_non_equity_etf"]))
                    or not holdings
                    or (h_count < 8 and total_weight < 0.85)
                ):
                    stale_or_missing = True
                else:
                    try:
                        updated_dt = ensure_utc(datetime.fromisoformat(str(meta["updated_at"])))
                        age_days = (now - updated_dt).total_seconds() / 86400.0
                        if age_days > max_holdings_staleness_days or age_days < -0.05:
                            stale_or_missing = True
                    except ValueError:
                        stale_or_missing = True

            if stale_or_missing:
                etf_frozen.add(etf_sym)
                has_warnings = True
                warn_msg = f"{etf_sym}: stale_or_missing_etf_holdings"
                warnings.append(warn_msg)
                etf_details[etf_sym] = EtfConcentrationInfo(
                    etf_symbol=etf_sym,
                    sector=entry.sector,
                    holdings_count=int(meta["holdings_count"]) if meta is not None else 0,
                    min_top_weight=float(meta["min_top_weight"]) if meta is not None else 0.0,
                    stale_or_missing=True,
                    frozen=True,
                    reason="stale_or_missing_etf_holdings",
                    updated_at=str(meta["updated_at"]) if meta is not None else None,
                )
                etf_weights_summary.append(f"{etf_sym}: STALE/MISSING holdings [FROZEN]")
                continue

        holdings_count = int(meta["holdings_count"]) if meta is not None else len(holdings)
        min_top_weight = (
            float(meta["min_top_weight"])
            if (meta is not None and float(meta["min_top_weight"]) > 0.0)
            else (min(holdings.values()) if holdings else 0.0)
        )
        unaccounted = max(0.0, 1.0 - sum(holdings.values()))

        best_observed = 0.0
        best_conservative = 0.0
        best_group: frozenset[str] = restricted_groups[0]

        for grp in restricted_groups:
            observed = sum(holdings.get(h, 0.0) for h in grp)
            canonical_group = {_canonical_share_class(c) for c in grp}
            present_canonical = {_canonical_share_class(h) for h in holdings if h in grp}
            missing_count = len(canonical_group) - len(present_canonical)
            missing_bound = min(missing_count * min_top_weight, unaccounted)
            if holdings_count >= 1 and (observed > 0.0 or holdings_count < 10):
                conservative = observed + missing_bound
            else:
                conservative = observed

            if (conservative, observed) > (best_conservative, best_observed):
                best_conservative = conservative
                best_observed = observed
                best_group = grp

        is_conc_frozen = best_conservative >= etf_concentration_cap
        is_direct_etf = etf_sym in direct_frozen
        frozen_flag = is_conc_frozen or is_direct_etf
        if is_conc_frozen:
            etf_frozen.add(etf_sym)

        reason = ""
        if is_direct_etf:
            reason = "direct_blackout_symbol"
        elif is_conc_frozen:
            reason = (
                f"etf_concentration_cap ({best_conservative:.4f} >= {etf_concentration_cap:.4f})"
            )

        etf_details[etf_sym] = EtfConcentrationInfo(
            etf_symbol=etf_sym,
            sector=entry.sector,
            observed_weight=round(best_observed, 6),
            conservative_weight=round(best_conservative, 6),
            holdings_count=holdings_count,
            min_top_weight=round(min_top_weight, 6),
            stale_or_missing=False,
            frozen=frozen_flag,
            reason=reason,
            updated_at=str(meta["updated_at"]) if meta is not None else None,
        )

        if best_observed > 0.0 or frozen_flag:
            grp_label = "/".join(sorted({_canonical_share_class(c) for c in best_group}))
            suffix = " [FROZEN]" if frozen_flag else ""
            etf_weights_summary.append(
                f"{etf_sym}: {best_conservative * 100.0:.1f}% {grp_label}{suffix}"
            )

    universe_symbols_set = {u.symbol for u in universe}
    all_frozen_set = set(direct_frozen) | etf_frozen
    for raw_obs in observed_broker_symbols:
        obs_sym = raw_obs.strip().upper()
        if not obs_sym:
            continue
        if obs_sym not in universe_symbols_set and obs_sym not in all_frozen_set:
            all_frozen_set.add(obs_sym)
            has_warnings = True
            warnings.append(f"{obs_sym}: out_of_universe_unverified_during_blackout")

    return BlackoutResolution(
        raw_blackout_period=raw,
        raw_symbols=raw_symbols,
        direct_frozen_symbols=direct_frozen,
        etf_frozen_symbols=frozenset(etf_frozen),
        frozen_symbols=frozenset(all_frozen_set),
        halted=False,
        halt_reason="",
        has_warnings=has_warnings,
        warnings=tuple(warnings),
        etf_details=etf_details,
        etf_weights_summary=etf_weights_summary,
    )


class EnvBlackoutTracker:
    """Stateful .env blackout resolver that fails closed if a seen/required .env disappears."""

    def __init__(
        self,
        *,
        env_file: Path | None = None,
        require_env_file: bool = False,
        fallback_raw: str = "",
    ) -> None:
        self.explicit_env_file: Path | None = Path(env_file) if env_file is not None else None
        self.require_env_file: bool = bool(require_env_file) or (env_file is not None)
        self.fallback_raw: str = fallback_raw
        self._saw_env_file: bool = (
            self.explicit_env_file is not None and self.explicit_env_file.exists()
        ) or (self.explicit_env_file is None and Path(".env").exists())

    def resolve(
        self,
        *,
        db: StateDB | None,
        config_dir: Path,
        now_utc: datetime | None = None,
        observed_broker_symbols: Iterable[str] = (),
    ) -> BlackoutResolution:
        universe, universe_error = load_universe_for_blackout(config_dir)
        if self.explicit_env_file is not None:
            return resolve_live_blackout(
                env_file=self.explicit_env_file,
                require_env_file=True,
                db=db,
                universe=universe,
                universe_error=universe_error,
                observed_broker_symbols=observed_broker_symbols,
                now_utc=now_utc,
            )

        default_path = Path(".env")
        if default_path.exists():
            self._saw_env_file = True
            return resolve_live_blackout(
                env_file=default_path,
                require_env_file=True,
                db=db,
                universe=universe,
                universe_error=universe_error,
                observed_broker_symbols=observed_broker_symbols,
                now_utc=now_utc,
            )

        if self.require_env_file or self._saw_env_file:
            return resolve_live_blackout(
                env_file=default_path,
                require_env_file=True,
                db=db,
                universe=universe,
                universe_error=universe_error,
                observed_broker_symbols=observed_broker_symbols,
                now_utc=now_utc,
            )

        return resolve_live_blackout(
            env_file=None,
            require_env_file=False,
            fallback_raw=self.fallback_raw,
            db=db,
            universe=universe,
            universe_error=universe_error,
            observed_broker_symbols=observed_broker_symbols,
            now_utc=now_utc,
        )
