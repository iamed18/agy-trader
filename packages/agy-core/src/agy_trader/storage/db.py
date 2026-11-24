from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from agy_trader.core.config import LLMQuotaConfig
from agy_trader.core.models import (
    ControlDials,
    EarningsEvent,
    LLMUsageRecord,
    MacroPoint,
    NewsArticle,
    SecFilingEvent,
    ensure_utc,
)
from agy_trader.data.news_filters import is_forward_only_source

_VALID_SENTIMENT_LABELS = frozenset({"positive", "negative", "neutral"})


def _dt_to_iso(dt: datetime) -> str:
    return ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _iso_to_dt(raw: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(raw))


def _coerce_ts_iso(ts: datetime | str | None) -> str:
    if ts is None:
        return _dt_to_iso(datetime.now(UTC))
    if isinstance(ts, datetime):
        return _dt_to_iso(ts)
    return _dt_to_iso(_iso_to_dt(ts))


class StateDB:
    """SQLite WAL-backed state and metadata store."""

    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def _connect(
        self,
        *,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = "DEFERRED",
    ) -> sqlite3.Connection:
        if self.read_only:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=30.0,
                isolation_level=isolation_level,
            )
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only = ON;")
                conn.execute("PRAGMA busy_timeout = 30000;")
            except Exception:
                conn.close()
                raise
            return conn
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=isolation_level,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
        except Exception:
            conn.close()
            raise
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript(
                """
                    CREATE TABLE IF NOT EXISTS news_articles (
                        article_id TEXT PRIMARY KEY,
                        published_at TEXT NOT NULL,
                        headline TEXT NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        source TEXT NOT NULL DEFAULT 'benzinga',
                        headline_hash TEXT NOT NULL UNIQUE,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        confidence REAL,
                        scorer_model TEXT,
                        scored_at TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_news_articles_published_at
                        ON news_articles (published_at);

                    CREATE TABLE IF NOT EXISTS news_article_symbols (
                        article_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        published_at TEXT NOT NULL,
                        PRIMARY KEY (article_id, symbol),
                        FOREIGN KEY (article_id)
                            REFERENCES news_articles (article_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_news_article_symbols_symbol_pub
                        ON news_article_symbols (symbol, published_at);

                    CREATE TABLE IF NOT EXISTS earnings_calendar (
                        symbol TEXT NOT NULL,
                        earnings_date TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (symbol, earnings_date)
                    );

                    CREATE TABLE IF NOT EXISTS sec_filings (
                        accession_number TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        cik TEXT NOT NULL,
                        form_type TEXT NOT NULL,
                        accepted_at TEXT NOT NULL,
                        items_json TEXT NOT NULL,
                        net_insider_buy_usd REAL NOT NULL,
                        insider_buyer_count INTEGER NOT NULL,
                        insider_seller_count INTEGER NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (accession_number, symbol)
                    );

                    CREATE INDEX IF NOT EXISTS idx_sec_filings_symbol_acc
                        ON sec_filings (symbol, accepted_at);

                    CREATE TABLE IF NOT EXISTS macro_snapshots (
                        series_id TEXT NOT NULL,
                        as_of_date TEXT NOT NULL,
                        value REAL NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (series_id, as_of_date)
                    );

                    CREATE TABLE IF NOT EXISTS control_dials (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        pause_new_entries INTEGER NOT NULL,
                        exposure_multiplier REAL NOT NULL,
                        allow_shorting INTEGER NOT NULL,
                        blacklisted_symbols_json TEXT NOT NULL,
                        updated_by TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS control_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        changed_at TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        dials_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS backtest_runs (
                        run_id TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS daemon_heartbeats (
                        daemon_name TEXT PRIMARY KEY,
                        last_heartbeat_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS llm_usage_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        tier INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        task TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        estimated_usd REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp
                        ON llm_usage_log (timestamp);

                    CREATE TABLE IF NOT EXISTS service_restart_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        service TEXT NOT NULL,
                        restarted_at TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        reason TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_service_restart_log_service_ts
                        ON service_restart_log (service, restarted_at);

                    CREATE TABLE IF NOT EXISTS live_orders (
                        client_order_id TEXT PRIMARY KEY,
                        broker_order_id TEXT,
                        run_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        qty REAL NOT NULL,
                        limit_price REAL,
                        stop_price REAL,
                        status TEXT NOT NULL,
                        filled_qty REAL NOT NULL DEFAULT 0.0,
                        filled_avg_price REAL,
                        reference_price REAL NOT NULL,
                        realized_slippage_bps REAL,
                        is_risk_reducing INTEGER NOT NULL,
                        submitted_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_live_orders_symbol_sub
                        ON live_orders (symbol, submitted_at);

                    CREATE INDEX IF NOT EXISTS idx_live_orders_status
                        ON live_orders (status);

                    CREATE TABLE IF NOT EXISTS live_positions (
                        symbol TEXT PRIMARY KEY,
                        qty REAL NOT NULL,
                        avg_entry_price REAL NOT NULL,
                        market_price REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        stop_order_id TEXT,
                        stop_price REAL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        equity REAL NOT NULL,
                        cash REAL NOT NULL,
                        buying_power REAL NOT NULL,
                        gross_exposure REAL NOT NULL,
                        net_exposure REAL NOT NULL,
                        daily_pnl_pct REAL NOT NULL,
                        drawdown_pct REAL NOT NULL,
                        high_water_mark REAL NOT NULL,
                        spy_close REAL,
                        ew50_index REAL
                    );

                    CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_ts
                        ON portfolio_snapshots (timestamp);

                    CREATE TABLE IF NOT EXISTS daemon_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        daemon_name TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_daemon_events_name_type_ts
                        ON daemon_events (daemon_name, event_type, timestamp);

                    CREATE INDEX IF NOT EXISTS idx_daemon_events_type_ts
                        ON daemon_events (event_type, timestamp, id);

                    CREATE TABLE IF NOT EXISTS etf_holdings (
                        etf_symbol TEXT NOT NULL,
                        holding_symbol TEXT NOT NULL,
                        weight REAL NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (etf_symbol, holding_symbol)
                    );

                    CREATE TABLE IF NOT EXISTS etf_holdings_meta (
                        etf_symbol TEXT PRIMARY KEY,
                        is_non_equity_etf INTEGER NOT NULL DEFAULT 0,
                        holdings_count INTEGER NOT NULL,
                        min_top_weight REAL NOT NULL DEFAULT 0.0,
                        updated_at TEXT NOT NULL
                    );
                    """
            )

    def _load_symbols_map(
        self, conn: sqlite3.Connection, article_ids: list[str]
    ) -> dict[str, list[str]]:
        if not article_ids:
            return {}
        symbols_map: dict[str, list[str]] = defaultdict(list)
        chunk_size = 500
        for idx in range(0, len(article_ids), chunk_size):
            chunk = article_ids[idx : idx + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT article_id, symbol
                FROM news_article_symbols
                WHERE article_id IN ({placeholders})
                ORDER BY symbol ASC
                """,
                chunk,
            ).fetchall()
            for row in rows:
                symbols_map[str(row["article_id"])].append(str(row["symbol"]))
        return symbols_map

    def _rows_to_news_articles(
        self, conn: sqlite3.Connection, rows: list[sqlite3.Row]
    ) -> list[NewsArticle]:
        article_ids = [str(row["article_id"]) for row in rows]
        symbols_map = self._load_symbols_map(conn, article_ids)
        articles: list[NewsArticle] = []
        for row in rows:
            aid = str(row["article_id"])
            raw_hash = str(row["headline_hash"])
            clean_hash = raw_hash.removeprefix("fwd:")
            articles.append(
                NewsArticle(
                    article_id=aid,
                    published_at=_iso_to_dt(str(row["published_at"])),
                    headline=str(row["headline"]),
                    summary=str(row["summary"]),
                    source=str(row["source"]),
                    symbols=symbols_map.get(aid, []),
                    headline_hash=clean_hash,
                    sentiment_score=(
                        float(row["sentiment_score"])
                        if row["sentiment_score"] is not None
                        else None
                    ),
                    sentiment_label=(
                        str(row["sentiment_label"])
                        if row["sentiment_label"] is not None
                        else None
                    ),
                    confidence=(
                        float(row["confidence"]) if row["confidence"] is not None else None
                    ),
                    scorer_model=(
                        str(row["scorer_model"]) if row["scorer_model"] is not None else None
                    ),
                )
            )
        return articles

    def upsert_news_articles(self, articles: list[NewsArticle]) -> int:
        if not articles:
            return 0
        now_iso = _dt_to_iso(datetime.now(UTC))
        inserted = 0

        with closing(self._connect()) as conn, conn:
            for article in articles:
                pub_iso = _dt_to_iso(article.published_at)
                scored_iso = now_iso if article.sentiment_score is not None else None
                db_hash = (
                    f"fwd:{article.headline_hash}"
                    if is_forward_only_source(article.source)
                    else article.headline_hash
                )
                cur = conn.execute(
                    """
                        INSERT OR IGNORE INTO news_articles (
                            article_id, published_at, headline, summary, source,
                            headline_hash, sentiment_score, sentiment_label,
                            confidence, scorer_model, scored_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    (
                        article.article_id,
                        pub_iso,
                        article.headline,
                        article.summary,
                        article.source,
                        db_hash,
                        article.sentiment_score,
                        article.sentiment_label,
                        article.confidence,
                        article.scorer_model,
                        scored_iso,
                        now_iso,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
                    canonical_id = article.article_id
                    canonical_pub_iso = pub_iso
                else:
                    row = conn.execute(
                        """
                        SELECT article_id, published_at
                        FROM news_articles
                        WHERE article_id = ? OR headline_hash = ?
                        LIMIT 1
                        """,
                        (article.article_id, db_hash),
                    ).fetchone()
                    if row is None:
                        continue
                    canonical_id = str(row["article_id"])
                    canonical_pub_iso = str(row["published_at"])

                unique_symbols = sorted(
                    {s.strip().upper() for s in article.symbols if s.strip()}
                )
                if unique_symbols:
                    conn.executemany(
                        """
                            INSERT OR IGNORE INTO news_article_symbols (
                                article_id, symbol, published_at
                            ) VALUES (?, ?, ?)
                            """,
                        [(canonical_id, sym, canonical_pub_iso) for sym in unique_symbols],
                    )
        return inserted

    def get_unscored_news(
        self, scorer_model: str | None = None, limit: int = 500
    ) -> list[NewsArticle]:
        with closing(self._connect()) as conn:
            if scorer_model is not None:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM news_articles
                    WHERE sentiment_score IS NULL
                       OR scorer_model IS NULL
                       OR scorer_model != ?
                    ORDER BY published_at ASC
                    LIMIT ?
                    """,
                    (scorer_model, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM news_articles
                    WHERE sentiment_score IS NULL
                    ORDER BY published_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return self._rows_to_news_articles(conn, rows)

    def update_news_scores(
        self, updates: list[tuple[str, float, str, float, str]]
    ) -> int:
        if not updates:
            return 0
        for article_id, score, label, confidence, scorer_model in updates:
            if not math.isfinite(score) or not (-1.0 <= score <= 1.0):
                raise ValueError(f"Invalid sentiment score for {article_id!r}: {score!r}")
            if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Invalid confidence for {article_id!r}: {confidence!r}")
            if label not in _VALID_SENTIMENT_LABELS:
                raise ValueError(f"Invalid sentiment label for {article_id!r}: {label!r}")
            if not scorer_model or not scorer_model.strip():
                raise ValueError(f"scorer_model must be non-empty for {article_id!r}")

        now_iso = _dt_to_iso(datetime.now(UTC))
        updated = 0
        with closing(self._connect()) as conn, conn:
            for article_id, score, label, confidence, scorer_model in updates:
                cur = conn.execute(
                    """
                        UPDATE news_articles
                        SET sentiment_score = ?,
                            sentiment_label = ?,
                            confidence = ?,
                            scorer_model = ?,
                            scored_at = ?
                        WHERE article_id = ?
                        """,
                    (score, label, confidence, scorer_model, now_iso, article_id),
                )
                updated += cur.rowcount
        return updated

    def get_news_for_symbol_window(
        self, symbol: str, start_utc: datetime, end_utc: datetime
    ) -> list[NewsArticle]:
        clean_symbol = symbol.strip().upper()
        start_iso = _dt_to_iso(start_utc)
        end_iso = _dt_to_iso(end_utc)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT na.*
                FROM news_article_symbols AS nas
                INNER JOIN news_articles AS na
                    ON na.article_id = nas.article_id
                WHERE nas.symbol = ?
                  AND nas.published_at >= ?
                  AND nas.published_at <= ?
                ORDER BY na.published_at ASC
                """,
                (clean_symbol, start_iso, end_iso),
            ).fetchall()
            return self._rows_to_news_articles(conn, rows)

    def get_all_scored_news(
        self,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        scorer_model: str | None = None,
        sources: Sequence[str] | None = None,
        include_forward_only: bool = False,
    ) -> list[NewsArticle]:
        clauses = ["sentiment_score IS NOT NULL"]
        params: list[Any] = []
        if not include_forward_only:
            clauses.append(
                "(LOWER(TRIM(source)) != 'yfinance_news' AND LOWER(TRIM(source)) NOT LIKE 'rss:%')"
            )
        if start_utc is not None:
            clauses.append("published_at >= ?")
            params.append(_dt_to_iso(start_utc))
        if end_utc is not None:
            clauses.append("published_at <= ?")
            params.append(_dt_to_iso(end_utc))
        if scorer_model is not None:
            clauses.append("scorer_model = ?")
            params.append(scorer_model)
        if sources is not None:
            clean_sources = [s.strip().lower() for s in sources if s.strip()]
            if not clean_sources:
                return []
            placeholders = ",".join("?" for _ in clean_sources)
            clauses.append(f"LOWER(source) IN ({placeholders})")
            params.extend(clean_sources)
        where_sql = " AND ".join(clauses)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM news_articles
                WHERE {where_sql}
                ORDER BY published_at ASC
                """,
                params,
            ).fetchall()
            return self._rows_to_news_articles(conn, rows)

    def upsert_earnings(self, events: list[EarningsEvent]) -> int:
        if not events:
            return 0
        now_iso = _dt_to_iso(datetime.now(UTC))
        deduped = {(ev.symbol, ev.earnings_date.isoformat()) for ev in events}
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                """
                    INSERT INTO earnings_calendar (symbol, earnings_date, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(symbol, earnings_date) DO UPDATE SET
                        updated_at = excluded.updated_at
                    """,
                [(sym, edate, now_iso) for sym, edate in deduped],
            )
        return len(deduped)

    def get_earnings_dates(self, symbol: str) -> list[date]:
        clean_symbol = symbol.strip().upper()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT earnings_date
                FROM earnings_calendar
                WHERE symbol = ?
                ORDER BY earnings_date ASC
                """,
                (clean_symbol,),
            ).fetchall()
        return [date.fromisoformat(str(row["earnings_date"])) for row in rows]

    def get_all_earnings_map(self) -> dict[str, list[date]]:
        result: dict[str, list[date]] = defaultdict(list)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT symbol, earnings_date
                FROM earnings_calendar
                ORDER BY symbol ASC, earnings_date ASC
                """
            ).fetchall()
        for row in rows:
            result[str(row["symbol"])].append(date.fromisoformat(str(row["earnings_date"])))
        return dict(result)

    def upsert_sec_filings(self, events: list[SecFilingEvent]) -> int:
        if not events:
            return 0
        now_iso = _dt_to_iso(datetime.now(UTC))
        deduped: dict[tuple[str, str], SecFilingEvent] = {
            (ev.accession_number, ev.symbol): ev for ev in events
        }
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                """
                INSERT INTO sec_filings (
                    accession_number, symbol, cik, form_type, accepted_at,
                    items_json, net_insider_buy_usd, insider_buyer_count,
                    insider_seller_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(accession_number, symbol) DO UPDATE SET
                    cik = excluded.cik,
                    form_type = excluded.form_type,
                    accepted_at = excluded.accepted_at,
                    items_json = excluded.items_json,
                    net_insider_buy_usd = excluded.net_insider_buy_usd,
                    insider_buyer_count = excluded.insider_buyer_count,
                    insider_seller_count = excluded.insider_seller_count,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        ev.accession_number,
                        ev.symbol,
                        ev.cik,
                        ev.form_type,
                        _dt_to_iso(ev.accepted_at),
                        json.dumps(ev.items),
                        float(ev.net_insider_buy_usd),
                        int(ev.insider_buyer_count),
                        int(ev.insider_seller_count),
                        now_iso,
                    )
                    for ev in deduped.values()
                ],
            )
        return len(deduped)

    def get_sec_filings_map(self) -> dict[str, list[SecFilingEvent]]:
        result: dict[str, list[SecFilingEvent]] = defaultdict(list)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT accession_number, symbol, cik, form_type, accepted_at,
                       items_json, net_insider_buy_usd, insider_buyer_count,
                       insider_seller_count
                FROM sec_filings
                ORDER BY symbol ASC, accepted_at ASC
                """
            ).fetchall()
        for row in rows:
            raw_form = str(row["form_type"])
            form_type: Literal["4", "8-K"] = "4" if raw_form == "4" else "8-K"
            ev = SecFilingEvent(
                accession_number=str(row["accession_number"]),
                symbol=str(row["symbol"]),
                cik=str(row["cik"]),
                form_type=form_type,
                accepted_at=_iso_to_dt(str(row["accepted_at"])),
                items=json.loads(str(row["items_json"])),
                net_insider_buy_usd=float(row["net_insider_buy_usd"]),
                insider_buyer_count=int(row["insider_buyer_count"]),
                insider_seller_count=int(row["insider_seller_count"]),
            )
            result[ev.symbol].append(ev)
        return dict(result)

    def upsert_macro(self, points: list[MacroPoint]) -> int:
        if not points:
            return 0
        now_iso = _dt_to_iso(datetime.now(UTC))
        deduped: dict[tuple[str, str], float] = {
            (pt.series_id, pt.as_of_date.isoformat()): pt.value for pt in points
        }
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                """
                    INSERT INTO macro_snapshots (series_id, as_of_date, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(series_id, as_of_date) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                [(sid, dt_str, val, now_iso) for (sid, dt_str), val in deduped.items()],
            )
        return len(deduped)

    def get_macro_series(
        self, series_id: str, up_to_date: date | None = None
    ) -> list[MacroPoint]:
        clean_series = series_id.strip()
        with closing(self._connect()) as conn:
            if up_to_date is not None:
                rows = conn.execute(
                    """
                    SELECT as_of_date, series_id, value
                    FROM macro_snapshots
                    WHERE series_id = ?
                      AND as_of_date <= ?
                    ORDER BY as_of_date ASC
                    """,
                    (clean_series, up_to_date.isoformat()),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT as_of_date, series_id, value
                    FROM macro_snapshots
                    WHERE series_id = ?
                    ORDER BY as_of_date ASC
                    """,
                    (clean_series,),
                ).fetchall()
        return [
            MacroPoint(
                as_of_date=date.fromisoformat(str(row["as_of_date"])),
                series_id=str(row["series_id"]),
                value=float(row["value"]),
            )
            for row in rows
        ]

    def get_control_dials(self) -> ControlDials:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM control_dials WHERE id = 1"
            ).fetchone()
        if row is None:
            return ControlDials()
        blacklist_raw = json.loads(str(row["blacklisted_symbols_json"]))
        return ControlDials(
            pause_new_entries=bool(row["pause_new_entries"]),
            exposure_multiplier=float(row["exposure_multiplier"]),
            allow_shorting=bool(row["allow_shorting"]),
            blacklisted_symbols=blacklist_raw,
            updated_by=str(row["updated_by"]),
            reason=str(row["reason"]),
            updated_at=_iso_to_dt(str(row["updated_at"])),
        )

    def update_control_dials(
        self, new_dials: ControlDials, actor: str, reason: str
    ) -> ControlDials:
        ts_utc = ensure_utc(new_dials.updated_at)
        validated = ControlDials.model_validate(
            {
                **new_dials.model_dump(),
                "updated_by": actor,
                "reason": reason,
                "updated_at": ts_utc,
            }
        )
        blacklist_json = json.dumps(
            {sym: _dt_to_iso(exp) for sym, exp in validated.blacklisted_symbols.items()},
            sort_keys=True,
        )
        ts_iso = _dt_to_iso(validated.updated_at)
        dials_audit_json = validated.model_dump_json()

        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                    INSERT INTO control_dials (
                        id, pause_new_entries, exposure_multiplier, allow_shorting,
                        blacklisted_symbols_json, updated_by, reason, updated_at
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        pause_new_entries = excluded.pause_new_entries,
                        exposure_multiplier = excluded.exposure_multiplier,
                        allow_shorting = excluded.allow_shorting,
                        blacklisted_symbols_json = excluded.blacklisted_symbols_json,
                        updated_by = excluded.updated_by,
                        reason = excluded.reason,
                        updated_at = excluded.updated_at
                    """,
                (
                    int(validated.pause_new_entries),
                    validated.exposure_multiplier,
                    int(validated.allow_shorting),
                    blacklist_json,
                    validated.updated_by,
                    validated.reason,
                    ts_iso,
                ),
            )
            conn.execute(
                """
                    INSERT INTO control_audit_log (changed_at, actor, reason, dials_json)
                    VALUES (?, ?, ?, ?)
                    """,
                (ts_iso, actor, reason, dials_audit_json),
            )
        return validated

    def get_control_audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, changed_at, actor, reason, dials_json
                FROM control_audit_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "changed_at": str(row["changed_at"]),
                "actor": str(row["actor"]),
                "reason": str(row["reason"]),
                "dials": json.loads(str(row["dials_json"])),
            }
            for row in rows
        ]

    def try_record_service_restart(
        self,
        service: str,
        max_per_hour: int,
        now_utc: datetime,
        actor: str,
        reason: str,
    ) -> bool:
        svc = service.strip().lower()
        clean_reason = reason.strip()
        now = ensure_utc(now_utc)
        now_iso = _dt_to_iso(now)
        cutoff_iso = _dt_to_iso(now - timedelta(hours=1))
        audit_reason = f"[RESTART:{svc}] {clean_reason}"

        with closing(self._connect(isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                count_row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM service_restart_log
                    WHERE service = ?
                      AND restarted_at >= ?
                    """,
                    (svc, cutoff_iso),
                ).fetchone()
                count = int(count_row["cnt"]) if count_row is not None else 0
                if count >= max_per_hour:
                    conn.execute("ROLLBACK")
                    return False

                dials_row = conn.execute(
                    "SELECT * FROM control_dials WHERE id = 1"
                ).fetchone()
                if dials_row is None:
                    current_dials = ControlDials()
                else:
                    current_dials = ControlDials(
                        pause_new_entries=bool(dials_row["pause_new_entries"]),
                        exposure_multiplier=float(dials_row["exposure_multiplier"]),
                        allow_shorting=bool(dials_row["allow_shorting"]),
                        blacklisted_symbols=json.loads(
                            str(dials_row["blacklisted_symbols_json"])
                        ),
                        updated_by=str(dials_row["updated_by"]),
                        reason=str(dials_row["reason"]),
                        updated_at=_iso_to_dt(str(dials_row["updated_at"])),
                    )
                audit_dials = current_dials.model_copy(
                    update={
                        "updated_by": actor,
                        "reason": audit_reason,
                        "updated_at": now,
                    }
                )

                conn.execute(
                    """
                    INSERT INTO service_restart_log (service, restarted_at, actor, reason)
                    VALUES (?, ?, ?, ?)
                    """,
                    (svc, now_iso, actor, clean_reason),
                )
                conn.execute(
                    """
                    INSERT INTO control_audit_log (changed_at, actor, reason, dials_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (now_iso, actor, audit_reason, audit_dials.model_dump_json()),
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def get_service_restart_log(
        self, service: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            if service is not None:
                svc = service.strip().lower()
                rows = conn.execute(
                    """
                    SELECT id, service, restarted_at, actor, reason
                    FROM service_restart_log
                    WHERE service = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (svc, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, service, restarted_at, actor, reason
                    FROM service_restart_log
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "service": str(row["service"]),
                "restarted_at": str(row["restarted_at"]),
                "actor": str(row["actor"]),
                "reason": str(row["reason"]),
            }
            for row in rows
        ]

    def log_llm_usage(self, record: LLMUsageRecord) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                    INSERT INTO llm_usage_log (
                        timestamp, tier, model, task,
                        prompt_tokens, completion_tokens, estimated_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    _dt_to_iso(record.timestamp),
                    record.tier,
                    record.model,
                    record.task,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.estimated_usd,
                ),
            )

    def get_llm_spend(self, since_utc: datetime) -> float:
        since_iso = _dt_to_iso(since_utc)
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(estimated_usd), 0.0) AS total_usd
                FROM llm_usage_log
                WHERE timestamp >= ?
                """,
                (since_iso,),
            ).fetchone()
        return float(row["total_usd"]) if row is not None else 0.0

    def check_llm_budget(
        self, quota: LLMQuotaConfig, now_utc: datetime
    ) -> tuple[bool, dict[str, float]]:
        now = ensure_utc(now_utc)
        day_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
        daily_spend = self.get_llm_spend(day_start)
        monthly_spend = self.get_llm_spend(month_start)
        allowed = (
            daily_spend < quota.daily_usd_cap
            and monthly_spend < quota.monthly_usd_cap
        )
        return allowed, {
            "daily_spend": daily_spend,
            "daily_cap": quota.daily_usd_cap,
            "monthly_spend": monthly_spend,
            "monthly_cap": quota.monthly_usd_cap,
        }

    def save_backtest_run(
        self,
        run_id: str,
        label: str,
        start_date: str,
        end_date: str,
        config_json: str,
        metrics_json: str,
    ) -> None:
        now_iso = _dt_to_iso(datetime.now(UTC))
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                    INSERT INTO backtest_runs (
                        run_id, label, start_date, end_date,
                        config_json, metrics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        label = excluded.label,
                        start_date = excluded.start_date,
                        end_date = excluded.end_date,
                        config_json = excluded.config_json,
                        metrics_json = excluded.metrics_json,
                        created_at = excluded.created_at
                    """,
                (run_id, label, start_date, end_date, config_json, metrics_json, now_iso),
            )

    def get_recent_backtest_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT run_id, label, start_date, end_date, config_json, metrics_json, created_at
                FROM backtest_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": str(row["run_id"]),
                "label": str(row["label"]),
                "start_date": str(row["start_date"]),
                "end_date": str(row["end_date"]),
                "config": json.loads(str(row["config_json"])),
                "metrics": json.loads(str(row["metrics_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def get_summary_counts(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            news_articles = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
            scored_news = conn.execute(
                "SELECT COUNT(*) FROM news_articles WHERE sentiment_score IS NOT NULL"
            ).fetchone()[0]
            earnings_events = conn.execute("SELECT COUNT(*) FROM earnings_calendar").fetchone()[0]
            sec_filings = conn.execute("SELECT COUNT(*) FROM sec_filings").fetchone()[0]
            macro_points = conn.execute("SELECT COUNT(*) FROM macro_snapshots").fetchone()[0]
            backtest_runs = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
        return {
            "news_articles": int(news_articles),
            "scored_news": int(scored_news),
            "earnings_events": int(earnings_events),
            "sec_filings": int(sec_filings),
            "macro_points": int(macro_points),
            "backtest_runs": int(backtest_runs),
        }

    # ------------------------------------------------------------------
    # Phase 2 Live-Execution & Daemon Operations
    # ------------------------------------------------------------------

    def check_wal_integrity(self) -> tuple[bool, str]:
        """Run PRAGMA quick_check to verify SQLite WAL database integrity."""
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute("PRAGMA quick_check;").fetchall()
        except sqlite3.DatabaseError as exc:
            return False, str(exc)
        messages = [str(row[0]) for row in rows if row]
        if messages == ["ok"]:
            return True, "ok"
        detail = "; ".join(messages) if messages else "empty quick_check result"
        return False, detail

    @staticmethod
    def _row_to_live_order(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "client_order_id": str(row["client_order_id"]),
            "broker_order_id": (
                str(row["broker_order_id"]) if row["broker_order_id"] is not None else None
            ),
            "run_id": str(row["run_id"]),
            "symbol": str(row["symbol"]),
            "side": str(row["side"]),
            "order_type": str(row["order_type"]),
            "qty": float(row["qty"]),
            "limit_price": (
                float(row["limit_price"]) if row["limit_price"] is not None else None
            ),
            "stop_price": float(row["stop_price"]) if row["stop_price"] is not None else None,
            "status": str(row["status"]),
            "filled_qty": float(row["filled_qty"]),
            "filled_avg_price": (
                float(row["filled_avg_price"]) if row["filled_avg_price"] is not None else None
            ),
            "reference_price": float(row["reference_price"]),
            "realized_slippage_bps": (
                float(row["realized_slippage_bps"])
                if row["realized_slippage_bps"] is not None
                else None
            ),
            "is_risk_reducing": bool(row["is_risk_reducing"]),
            "submitted_at": str(row["submitted_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_live_order(
        self,
        *,
        client_order_id: str,
        run_id: str,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        status: str,
        reference_price: float,
        is_risk_reducing: bool,
        broker_order_id: str | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        filled_qty: float = 0.0,
        filled_avg_price: float | None = None,
        realized_slippage_bps: float | None = None,
        submitted_at: datetime | str | None = None,
        updated_at: datetime | str | None = None,
    ) -> None:
        clean_cid = client_order_id.strip()
        if not clean_cid:
            raise ValueError("client_order_id must not be empty")
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            raise ValueError("symbol must not be empty")
        sub_iso = _coerce_ts_iso(submitted_at)
        upd_iso = _coerce_ts_iso(updated_at)

        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO live_orders (
                    client_order_id, broker_order_id, run_id, symbol, side,
                    order_type, qty, limit_price, stop_price, status,
                    filled_qty, filled_avg_price, reference_price,
                    realized_slippage_bps, is_risk_reducing, submitted_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    broker_order_id = COALESCE(excluded.broker_order_id, live_orders.broker_order_id),
                    status = excluded.status,
                    limit_price = COALESCE(excluded.limit_price, live_orders.limit_price),
                    stop_price = COALESCE(excluded.stop_price, live_orders.stop_price),
                    filled_qty = excluded.filled_qty,
                    filled_avg_price = COALESCE(excluded.filled_avg_price, live_orders.filled_avg_price),
                    realized_slippage_bps = COALESCE(
                        excluded.realized_slippage_bps, live_orders.realized_slippage_bps
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    clean_cid,
                    broker_order_id,
                    run_id,
                    clean_sym,
                    side.strip().lower(),
                    order_type.strip().lower(),
                    float(qty),
                    float(limit_price) if limit_price is not None else None,
                    float(stop_price) if stop_price is not None else None,
                    status.strip().lower(),
                    float(filled_qty),
                    float(filled_avg_price) if filled_avg_price is not None else None,
                    float(reference_price),
                    (
                        float(realized_slippage_bps)
                        if realized_slippage_bps is not None
                        else None
                    ),
                    int(bool(is_risk_reducing)),
                    sub_iso,
                    upd_iso,
                ),
            )

    @staticmethod
    def _is_missing_table(exc: sqlite3.OperationalError) -> bool:
        return "no such table" in str(exc).lower()

    def get_live_order(self, client_order_id: str) -> dict[str, Any] | None:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT * FROM live_orders WHERE client_order_id = ?",
                    (client_order_id.strip(),),
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return None
            raise
        if row is None:
            return None
        return self._row_to_live_order(row)

    def get_recent_live_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM live_orders
                    ORDER BY submitted_at DESC, rowid DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return []
            raise
        return [self._row_to_live_order(row) for row in rows]

    def get_open_live_orders(self) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM live_orders
                    WHERE status IN (
                        'new', 'accepted', 'partially_filled', 'pending_new',
                        'pending_cancel', 'pending_replace', 'held', 'stopped',
                        'accepted_for_bidding', 'calculated'
                    )
                    ORDER BY submitted_at ASC, rowid ASC
                    """
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return []
            raise
        return [self._row_to_live_order(row) for row in rows]

    def replace_live_positions(self, positions: Sequence[dict[str, Any]]) -> int:
        now_iso = _dt_to_iso(datetime.now(UTC))
        rows_to_insert: list[tuple[Any, ...]] = []
        for pos in positions:
            sym = str(pos["symbol"]).strip().upper()
            if not sym:
                raise ValueError("Position symbol must not be empty")
            upd_iso = _coerce_ts_iso(pos.get("updated_at")) if pos.get("updated_at") else now_iso
            stop_px_raw = pos.get("stop_price")
            rows_to_insert.append(
                (
                    sym,
                    float(pos["qty"]),
                    float(pos["avg_entry_price"]),
                    float(pos["market_price"]),
                    float(pos["unrealized_pnl"]),
                    str(pos["stop_order_id"]) if pos.get("stop_order_id") is not None else None,
                    float(stop_px_raw) if stop_px_raw is not None else None,
                    upd_iso,
                )
            )

        with closing(self._connect(isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM live_positions")
                if rows_to_insert:
                    conn.executemany(
                        """
                        INSERT INTO live_positions (
                            symbol, qty, avg_entry_price, market_price,
                            unrealized_pnl, stop_order_id, stop_price, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows_to_insert,
                    )
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return len(rows_to_insert)

    def get_live_positions(self) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT * FROM live_positions ORDER BY symbol ASC"
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return []
            raise
        return [
            {
                "symbol": str(row["symbol"]),
                "qty": float(row["qty"]),
                "avg_entry_price": float(row["avg_entry_price"]),
                "market_price": float(row["market_price"]),
                "unrealized_pnl": float(row["unrealized_pnl"]),
                "stop_order_id": (
                    str(row["stop_order_id"]) if row["stop_order_id"] is not None else None
                ),
                "stop_price": (
                    float(row["stop_price"]) if row["stop_price"] is not None else None
                ),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "timestamp": str(row["timestamp"]),
            "equity": float(row["equity"]),
            "cash": float(row["cash"]),
            "buying_power": float(row["buying_power"]),
            "gross_exposure": float(row["gross_exposure"]),
            "net_exposure": float(row["net_exposure"]),
            "daily_pnl_pct": float(row["daily_pnl_pct"]),
            "drawdown_pct": float(row["drawdown_pct"]),
            "high_water_mark": float(row["high_water_mark"]),
            "spy_close": float(row["spy_close"]) if row["spy_close"] is not None else None,
            "ew50_index": float(row["ew50_index"]) if row["ew50_index"] is not None else None,
        }

    def insert_portfolio_snapshot(
        self,
        *,
        timestamp: datetime | str | None = None,
        equity: float,
        cash: float,
        buying_power: float,
        gross_exposure: float,
        net_exposure: float,
        daily_pnl_pct: float,
        drawdown_pct: float,
        high_water_mark: float,
        spy_close: float | None = None,
        ew50_index: float | None = None,
    ) -> int:
        required = (
            equity,
            cash,
            buying_power,
            gross_exposure,
            net_exposure,
            daily_pnl_pct,
            drawdown_pct,
            high_water_mark,
        )
        if not all(math.isfinite(float(v)) for v in required):
            raise ValueError("Portfolio snapshot numeric fields must be finite")
        ts_iso = _coerce_ts_iso(timestamp)
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO portfolio_snapshots (
                    timestamp, equity, cash, buying_power, gross_exposure,
                    net_exposure, daily_pnl_pct, drawdown_pct, high_water_mark,
                    spy_close, ew50_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    float(equity),
                    float(cash),
                    float(buying_power),
                    float(gross_exposure),
                    float(net_exposure),
                    float(daily_pnl_pct),
                    float(drawdown_pct),
                    float(high_water_mark),
                    float(spy_close) if spy_close is not None else None,
                    float(ew50_index) if ew50_index is not None else None,
                ),
            )
            return int(cur.lastrowid or 0)

    def get_portfolio_snapshots(self, limit: int = 5000) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT *
                        FROM portfolio_snapshots
                        ORDER BY timestamp DESC, id DESC
                        LIMIT ?
                    )
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return []
            raise
        return [self._row_to_snapshot(row) for row in rows]

    def get_first_portfolio_snapshot(self) -> dict[str, Any] | None:
        """Return the earliest portfolio snapshot in O(1) via primary key."""
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM portfolio_snapshots
                    ORDER BY id ASC
                    LIMIT 1
                    """
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return None
            raise
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def get_first_invested_portfolio_snapshot(self) -> dict[str, Any] | None:
        """Return the earliest portfolio snapshot with non-zero gross exposure."""
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM portfolio_snapshots
                    WHERE gross_exposure > 0.0
                    ORDER BY timestamp ASC, id ASC
                    LIMIT 1
                    """
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return None
            raise
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def get_latest_portfolio_snapshot(self) -> dict[str, Any] | None:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM portfolio_snapshots
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 1
                    """
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return None
            raise
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def record_daemon_event(
        self,
        daemon_name: str,
        event_type: str,
        details: dict[str, Any] | None = None,
        timestamp: datetime | str | None = None,
    ) -> int:
        clean_daemon = daemon_name.strip()
        clean_type = event_type.strip().lower()
        if not clean_daemon or not clean_type:
            raise ValueError("daemon_name and event_type must be non-empty")
        ts_iso = _coerce_ts_iso(timestamp)
        details_json = json.dumps(details or {}, sort_keys=True, default=str)
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO daemon_events (timestamp, daemon_name, event_type, details_json)
                VALUES (?, ?, ?, ?)
                """,
                (ts_iso, clean_daemon, clean_type, details_json),
            )
            return int(cur.lastrowid or 0)

    def get_daemon_events(
        self,
        daemon_name: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if daemon_name is not None:
            clauses.append("daemon_name = ?")
            params.append(daemon_name.strip())
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type.strip().lower())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    f"""
                    SELECT id, timestamp, daemon_name, event_type, details_json
                    FROM daemon_events
                    {where_sql}
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return []
            raise
        return [
            {
                "id": int(row["id"]),
                "timestamp": str(row["timestamp"]),
                "daemon_name": str(row["daemon_name"]),
                "event_type": str(row["event_type"]),
                "details": json.loads(str(row["details_json"])),
            }
            for row in rows
        ]

    def has_session_rebalance_completed(self, session_date: str, checkpoint: str) -> bool:
        target_date = session_date.strip()
        target_ckpt = checkpoint.strip()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT details_json
                FROM daemon_events
                WHERE event_type = 'rebalance'
                ORDER BY timestamp DESC, id DESC
                """
            ).fetchall()
        for row in rows:
            details = json.loads(str(row["details_json"]))
            if (
                str(details.get("session_date", "")) == target_date
                and str(details.get("checkpoint", "")) == target_ckpt
            ):
                status = details.get("status")
                if status is None or str(status).lower() in {"completed", "ok", "success"}:
                    return True
        return False

    def has_month_rebalance_completed(
        self,
        year: int,
        month: int,
        *,
        daemon_name: str = "trader",
        exclude_session_date: str | None = None,
    ) -> bool:
        clean_daemon = daemon_name.strip()
        excl = exclude_session_date.strip() if exclude_session_date else None
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT details_json
                    FROM daemon_events
                    WHERE daemon_name = ? AND event_type = 'rebalance'
                    ORDER BY timestamp DESC, id DESC
                    """,
                    (clean_daemon,),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return False
            raise

        for row in rows:
            details = json.loads(str(row["details_json"]))
            status = details.get("status")
            if status is not None and str(status).lower() not in {"completed", "ok", "success"}:
                continue
            if bool(details.get("paused", False)):
                continue
            if (
                int(details.get("proposed_count", 0)) > 0
                and int(details.get("approved_count", 0)) == 0
            ):
                continue
            last_session = str(details.get("session_date", "")).strip()
            if not last_session or (excl is not None and last_session == excl):
                continue
            try:
                last_dt = datetime.fromisoformat(f"{last_session}T00:00:00").date()
            except ValueError:
                continue
            if last_dt.year == year and last_dt.month == month:
                return True
        return False

    def prune_old_telemetry(
        self,
        keep_days: int = 90,
        *,
        now_utc: datetime | None = None,
    ) -> dict[str, int]:
        """Prune old non-rebalance daemon_events and old portfolio_snapshots."""
        if keep_days < 1:
            raise ValueError("keep_days must be >= 1")
        ref_now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        cutoff_iso = _dt_to_iso(ref_now - timedelta(days=keep_days))
        with closing(self._connect()) as conn:
            with conn:
                ev_cur = conn.execute(
                    """
                    DELETE FROM daemon_events
                    WHERE timestamp < ?
                      AND event_type != 'rebalance'
                    """,
                    (cutoff_iso,),
                )
                snap_cur = conn.execute(
                    """
                    DELETE FROM portfolio_snapshots
                    WHERE timestamp < ?
                      AND id != (SELECT MIN(id) FROM portfolio_snapshots)
                      AND id != COALESCE(
                          (
                              SELECT id
                              FROM portfolio_snapshots
                              WHERE gross_exposure > 0.0
                              ORDER BY timestamp ASC, id ASC
                              LIMIT 1
                          ),
                          -1
                      )
                    """,
                    (cutoff_iso,),
                )
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        return {
            "daemon_events": int(ev_cur.rowcount),
            "portfolio_snapshots": int(snap_cur.rowcount),
        }

    def upsert_daemon_heartbeat(
        self,
        daemon_name: str,
        status: str,
        details: dict[str, Any] | None = None,
        now_utc: datetime | None = None,
    ) -> None:
        clean_daemon = daemon_name.strip()
        clean_status = status.strip().lower()
        if not clean_daemon or not clean_status:
            raise ValueError("daemon_name and status must be non-empty")
        ts_iso = _dt_to_iso(now_utc) if now_utc is not None else _dt_to_iso(datetime.now(UTC))
        details_json = json.dumps(details or {}, sort_keys=True, default=str)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO daemon_heartbeats (
                    daemon_name, last_heartbeat_at, status, details_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(daemon_name) DO UPDATE SET
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    status = excluded.status,
                    details_json = excluded.details_json
                """,
                (clean_daemon, ts_iso, clean_status, details_json),
            )

    def get_daemon_heartbeats(self) -> dict[str, dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT daemon_name, last_heartbeat_at, status, details_json
                FROM daemon_heartbeats
                ORDER BY daemon_name ASC
                """
            ).fetchall()
        return {
            str(row["daemon_name"]): {
                "daemon_name": str(row["daemon_name"]),
                "last_heartbeat_at": str(row["last_heartbeat_at"]),
                "status": str(row["status"]),
                "details": json.loads(str(row["details_json"])),
            }
            for row in rows
        }

    def count_unscored_news(self, scorer_model: str | None = None) -> int:
        """Return the exact count of unscored (or mismatched-model) news articles."""
        with closing(self._connect()) as conn:
            if scorer_model is not None:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM news_articles
                    WHERE sentiment_score IS NULL
                       OR scorer_model IS NULL
                       OR scorer_model != ?
                    """,
                    (scorer_model,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM news_articles WHERE sentiment_score IS NULL"
                ).fetchone()
        return int(row["cnt"]) if row is not None else 0

    def get_latest_news_timestamp(self, sources: Sequence[str] | None = None) -> datetime | None:
        """Return the most recent published_at timestamp in news_articles, or None."""
        clauses: list[str] = []
        params: list[Any] = []
        if sources is not None:
            clean = [s.strip().lower() for s in sources if s.strip()]
            if not clean:
                return None
            placeholders = ",".join("?" for _ in clean)
            clauses.append(f"LOWER(source) IN ({placeholders})")
            params.extend(clean)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT MAX(published_at) AS max_ts FROM news_articles {where_sql}",
                params,
            ).fetchone()
        if row is None or row["max_ts"] is None:
            return None
        return _iso_to_dt(str(row["max_ts"]))

    def get_recent_news(
        self,
        *,
        limit: int = 50,
        forward_only: bool | None = None,
        start_utc: datetime | None = None,
        scorer_model: str | None = None,
    ) -> list[NewsArticle]:
        """Return bounded recent news articles ordered newest-first."""
        clauses: list[str] = []
        params: list[Any] = []
        if forward_only is True:
            clauses.append(
                "(LOWER(TRIM(source)) = 'yfinance_news' OR LOWER(TRIM(source)) LIKE 'rss:%')"
            )
        elif forward_only is False:
            clauses.append(
                "(LOWER(TRIM(source)) != 'yfinance_news' AND LOWER(TRIM(source)) NOT LIKE 'rss:%')"
            )
        if start_utc is not None:
            clauses.append("published_at >= ?")
            params.append(_dt_to_iso(start_utc))
        if scorer_model is not None:
            clauses.append("scorer_model = ?")
            params.append(scorer_model)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM news_articles
                {where_sql}
                ORDER BY published_at DESC, rowid DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return self._rows_to_news_articles(conn, rows)

    def get_recent_sec_filings(
        self,
        *,
        limit: int = 50,
        start_utc: datetime | None = None,
    ) -> list[SecFilingEvent]:
        """Return bounded recent SEC Form 4 / 8-K filings ordered newest-first."""
        clauses: list[str] = []
        params: list[Any] = []
        if start_utc is not None:
            clauses.append("accepted_at >= ?")
            params.append(_dt_to_iso(start_utc))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT accession_number, symbol, cik, form_type, accepted_at,
                       items_json, net_insider_buy_usd, insider_buyer_count,
                       insider_seller_count
                FROM sec_filings
                {where_sql}
                ORDER BY accepted_at DESC, rowid DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        events: list[SecFilingEvent] = []
        for row in rows:
            raw_form = str(row["form_type"])
            form_type: Literal["4", "8-K"] = "4" if raw_form == "4" else "8-K"
            events.append(
                SecFilingEvent(
                    accession_number=str(row["accession_number"]),
                    symbol=str(row["symbol"]),
                    cik=str(row["cik"]),
                    form_type=form_type,
                    accepted_at=_iso_to_dt(str(row["accepted_at"])),
                    items=json.loads(str(row["items_json"])),
                    net_insider_buy_usd=float(row["net_insider_buy_usd"]),
                    insider_buyer_count=int(row["insider_buyer_count"]),
                    insider_seller_count=int(row["insider_seller_count"]),
                )
            )
        return events

    def get_latest_macro_points(
        self, series_ids: Sequence[str] | None = None
    ) -> dict[str, MacroPoint]:
        """Return the latest MacroPoint per series_id."""
        clauses: list[str] = []
        params: list[Any] = []
        if series_ids is not None:
            clean = [s.strip() for s in series_ids if s.strip()]
            if not clean:
                return {}
            placeholders = ",".join("?" for _ in clean)
            clauses.append(f"series_id IN ({placeholders})")
            params.extend(clean)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT m.series_id, m.as_of_date, m.value
                FROM macro_snapshots AS m
                INNER JOIN (
                    SELECT series_id, MAX(as_of_date) AS max_date
                    FROM macro_snapshots
                    {where_sql}
                    GROUP BY series_id
                ) AS latest
                    ON m.series_id = latest.series_id AND m.as_of_date = latest.max_date
                ORDER BY m.series_id ASC
                """,
                params,
            ).fetchall()
        return {
            str(row["series_id"]): MacroPoint(
                as_of_date=date.fromisoformat(str(row["as_of_date"])),
                series_id=str(row["series_id"]),
                value=float(row["value"]),
            )
            for row in rows
        }

    def upsert_etf_holdings(
        self,
        etf_symbol: str,
        holdings: Mapping[str, float],
        *,
        is_non_equity_etf: bool = False,
        updated_at: datetime | str | None = None,
    ) -> int:
        """Replace top holdings for an ETF and record refresh metadata."""
        clean_etf = etf_symbol.strip().upper()
        if not clean_etf:
            raise ValueError("etf_symbol must not be empty")

        clean_holdings: dict[str, float] = {}
        for raw_sym, raw_weight in holdings.items():
            sym = str(raw_sym).strip().upper()
            if not sym:
                continue
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
                raise ValueError(
                    f"Invalid ETF holding weight for {clean_etf}/{sym}: {raw_weight!r}"
                )
            clean_holdings[sym] = weight

        if sum(clean_holdings.values()) > 1.05:
            raise ValueError(
                f"ETF holdings weight sum {sum(clean_holdings.values()):.4f} exceeds 1.05 for {clean_etf}"
            )

        upd_iso = _coerce_ts_iso(updated_at)
        holdings_count = len(clean_holdings)
        min_top_weight = min(clean_holdings.values()) if clean_holdings else 0.0

        with closing(self._connect(isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM etf_holdings WHERE etf_symbol = ?", (clean_etf,))
                if clean_holdings:
                    conn.executemany(
                        """
                        INSERT INTO etf_holdings (
                            etf_symbol, holding_symbol, weight, updated_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        [
                            (clean_etf, holding_sym, weight, upd_iso)
                            for holding_sym, weight in sorted(clean_holdings.items())
                        ],
                    )
                conn.execute(
                    """
                    INSERT INTO etf_holdings_meta (
                        etf_symbol, is_non_equity_etf, holdings_count, min_top_weight, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(etf_symbol) DO UPDATE SET
                        is_non_equity_etf = excluded.is_non_equity_etf,
                        holdings_count = excluded.holdings_count,
                        min_top_weight = excluded.min_top_weight,
                        updated_at = excluded.updated_at
                    """,
                    (
                        clean_etf,
                        int(bool(is_non_equity_etf)),
                        holdings_count,
                        float(min_top_weight),
                        upd_iso,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return holdings_count

    def get_etf_holdings(self, etf_symbol: str) -> dict[str, float]:
        """Return stored top-holding weights keyed by uppercase holding symbol."""
        clean_etf = etf_symbol.strip().upper()
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT holding_symbol, weight
                    FROM etf_holdings
                    WHERE etf_symbol = ?
                    ORDER BY weight DESC, holding_symbol ASC
                    """,
                    (clean_etf,),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return {}
            raise
        return {str(row["holding_symbol"]): float(row["weight"]) for row in rows}

    def get_etf_holdings_meta(self, etf_symbol: str) -> dict[str, Any] | None:
        """Return refresh metadata for a single ETF, or None if not cached."""
        clean_etf = etf_symbol.strip().upper()
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    SELECT etf_symbol, is_non_equity_etf, holdings_count, min_top_weight, updated_at
                    FROM etf_holdings_meta
                    WHERE etf_symbol = ?
                    """,
                    (clean_etf,),
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return None
            raise
        if row is None:
            return None
        return {
            "etf_symbol": str(row["etf_symbol"]),
            "is_non_equity_etf": bool(row["is_non_equity_etf"]),
            "holdings_count": int(row["holdings_count"]),
            "min_top_weight": float(row["min_top_weight"]),
            "updated_at": str(row["updated_at"]),
        }

    def get_all_etf_holdings_meta(self) -> dict[str, dict[str, Any]]:
        """Return refresh metadata for all cached ETFs keyed by uppercase ETF symbol."""
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT etf_symbol, is_non_equity_etf, holdings_count, min_top_weight, updated_at
                    FROM etf_holdings_meta
                    ORDER BY etf_symbol ASC
                    """
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if self._is_missing_table(exc):
                return {}
            raise
        return {
            str(row["etf_symbol"]): {
                "etf_symbol": str(row["etf_symbol"]),
                "is_non_equity_etf": bool(row["is_non_equity_etf"]),
                "holdings_count": int(row["holdings_count"]),
                "min_top_weight": float(row["min_top_weight"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        }
