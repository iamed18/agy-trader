from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
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
    ensure_utc,
)

_VALID_SENTIMENT_LABELS = frozenset({"positive", "negative", "neutral"})


def _dt_to_iso(dt: datetime) -> str:
    return ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _iso_to_dt(raw: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(raw))


class StateDB:
    """SQLite WAL-backed state and metadata store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(
        self,
        *,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = "DEFERRED",
    ) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=isolation_level,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
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
            articles.append(
                NewsArticle(
                    article_id=aid,
                    published_at=_iso_to_dt(str(row["published_at"])),
                    headline=str(row["headline"]),
                    summary=str(row["summary"]),
                    source=str(row["source"]),
                    symbols=symbols_map.get(aid, []),
                    headline_hash=str(row["headline_hash"]),
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
                        article.headline_hash,
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
                        (article.article_id, article.headline_hash),
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
    ) -> list[NewsArticle]:
        clauses = ["sentiment_score IS NOT NULL"]
        params: list[Any] = []
        if start_utc is not None:
            clauses.append("published_at >= ?")
            params.append(_dt_to_iso(start_utc))
        if end_utc is not None:
            clauses.append("published_at <= ?")
            params.append(_dt_to_iso(end_utc))
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
            macro_points = conn.execute("SELECT COUNT(*) FROM macro_snapshots").fetchone()[0]
            backtest_runs = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
        return {
            "news_articles": int(news_articles),
            "scored_news": int(scored_news),
            "earnings_events": int(earnings_events),
            "macro_points": int(macro_points),
            "backtest_runs": int(backtest_runs),
        }
