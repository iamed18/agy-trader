from __future__ import annotations

import json
import logging
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import httpx

from agy_trader.core.models import SecFilingEvent, ensure_utc
from agy_trader.data.news_filters import SYMBOL_CANONICAL_MAP, SYMBOL_QUERY_ALIASES
from agy_trader.data.providers import SecFilingsProvider

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_TRUE_FLAGS = frozenset({"1", "true", "yes", "y"})
_TICKERS_CACHE_MAX_AGE_SECONDS = 7 * 86400


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem:
        if _local_name(child.tag) == name:
            return child
    return None


def _iter_descendants(elem: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in elem.iter() if _local_name(node.tag) == name]


def _read_value_text(parent: ET.Element | None, child_name: str) -> str:
    if parent is None:
        return ""
    node = _find_child(parent, child_name)
    if node is None:
        return ""
    val_node = _find_child(node, "value")
    target = val_node if val_node is not None else node
    return (target.text or "").strip()


def _parse_float_header(response: httpx.Response, name: str) -> float | None:
    raw = response.headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


def _parse_sec_acceptance_dt(raw_acc_dt: Any, raw_filing_date: Any) -> datetime | None:
    if raw_acc_dt is not None:
        cleaned = str(raw_acc_dt).strip()
        if cleaned:
            if cleaned.isdigit() and len(cleaned) == 14:
                try:
                    return (
                        datetime.strptime(cleaned, "%Y%m%d%H%M%S")
                        .replace(tzinfo=NY_TZ)
                        .astimezone(UTC)
                    )
                except ValueError:
                    pass
            iso_candidate = f"{cleaned[:-1]}+00:00" if cleaned.endswith("Z") else cleaned
            try:
                parsed = datetime.fromisoformat(iso_candidate)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    parsed = parsed.replace(tzinfo=NY_TZ)
                return ensure_utc(parsed)
            except ValueError:
                pass

    if raw_filing_date is not None:
        fd_str = str(raw_filing_date).strip()
        if fd_str:
            try:
                fd = date.fromisoformat(fd_str[:10])
                # Conservative post-close/Reg S-T cutoff fallback (22:05 ET) when acceptanceDateTime is absent
                return datetime.combine(fd, dt_time(22, 5), tzinfo=NY_TZ).astimezone(UTC)
            except ValueError:
                return None
    return None


def _normalize_items(raw_items: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_items, str):
        candidates = [part.strip() for part in raw_items.split(",")]
    elif isinstance(raw_items, list):
        candidates = [str(part).strip() for part in raw_items if part is not None]
    else:
        candidates = []

    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _parse_form4_xml(xml_text: str) -> tuple[float, int, int]:
    root = ET.fromstring(xml_text)

    qualifying_owners: set[str] = set()
    for owner_idx, owner_el in enumerate(_iter_descendants(root, "reportingOwner")):
        rel_el = _find_child(owner_el, "reportingOwnerRelationship")
        if rel_el is None:
            continue
        is_officer = _read_value_text(rel_el, "isOfficer").lower() in _TRUE_FLAGS
        is_director = _read_value_text(rel_el, "isDirector").lower() in _TRUE_FLAGS
        if not (is_officer or is_director):
            continue

        id_el = _find_child(owner_el, "reportingOwnerId")
        owner_cik = _read_value_text(id_el, "rptOwnerCik")
        owner_name = _read_value_text(id_el, "rptOwnerName")
        owner_key = owner_cik or owner_name or f"owner-{owner_idx}"
        qualifying_owners.add(owner_key.upper())

    if not qualifying_owners:
        return 0.0, 0, 0

    owner_count = len(qualifying_owners)
    buy_usd = 0.0
    sell_usd = 0.0
    has_buy = False
    has_sell = False

    for tx_el in _iter_descendants(root, "nonDerivativeTransaction"):
        coding_el = _find_child(tx_el, "transactionCoding")
        tx_code = _read_value_text(coding_el, "transactionCode").upper()
        if tx_code not in {"P", "S"}:
            continue

        amounts_el = _find_child(tx_el, "transactionAmounts")
        shares_raw = _read_value_text(amounts_el, "transactionShares")
        price_raw = _read_value_text(amounts_el, "transactionPricePerShare")
        if not shares_raw or not price_raw:
            continue
        try:
            shares = float(shares_raw)
            price = float(price_raw)
        except ValueError:
            continue
        if not (math.isfinite(shares) and math.isfinite(price)) or shares <= 0.0 or price <= 0.0:
            continue

        usd = shares * price
        if not math.isfinite(usd):
            continue
        if tx_code == "P":
            buy_usd += usd
            has_buy = True
        else:
            sell_usd += usd
            has_sell = True

    net_buy_usd = buy_usd - sell_usd
    if not math.isfinite(net_buy_usd):
        return 0.0, 0, 0
    buyer_count = owner_count if has_buy else 0
    seller_count = owner_count if has_sell else 0
    return net_buy_usd, buyer_count, seller_count


class SecEdgarProvider(SecFilingsProvider):
    """Point-in-time SEC EDGAR Form 4 & Form 8-K provider."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
        client: httpx.Client | None = None,
        min_interval_seconds: float = 0.13,
        max_form4_per_symbol: int = 200,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
        tickers_url: str = "https://www.sec.gov/files/company_tickers.json",
        submissions_base_url: str = "https://data.sec.gov/submissions",
        archives_base_url: str = "https://www.sec.gov/Archives/edgar/data",
    ) -> None:
        cleaned_ua = user_agent.strip()
        if not cleaned_ua:
            raise ValueError(
                "SecEdgarProvider requires a non-empty descriptive user_agent "
                "(e.g. 'agy-trader ops@example.com')."
            )
        if min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be >= 0")
        if max_form4_per_symbol < 0:
            raise ValueError("max_form4_per_symbol must be >= 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.user_agent = cleaned_ua
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._external_client = client
        self.min_interval_seconds = min_interval_seconds
        self.max_form4_per_symbol = max_form4_per_symbol
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep_fn
        self._now = now_fn
        self._last_request_ts: float | None = None
        self.tickers_url = tickers_url
        self.submissions_base_url = submissions_base_url.rstrip("/")
        self.archives_base_url = archives_base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _pace(self) -> None:
        if self.min_interval_seconds <= 0.0 or self._last_request_ts is None:
            return
        elapsed = self._now() - self._last_request_ts
        if elapsed < self.min_interval_seconds:
            self._sleep(self.min_interval_seconds - elapsed)

    def _compute_retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = _parse_float_header(response, "Retry-After")
        if retry_after is not None and retry_after > 0:
            return min(self._max_backoff_seconds, retry_after)
        return min(self._max_backoff_seconds, self._backoff_base_seconds * (2**attempt))

    def _get(self, client: httpx.Client, url: str) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._pace()
            self._last_request_ts = self._now()
            response = client.get(url, headers=self._headers())
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt >= self._max_retries:
                response.raise_for_status()
            delay = self._compute_retry_delay(response, attempt)
            logger.warning(
                "SEC EDGAR HTTP %s for %s (attempt %d/%d); retrying in %.2fs",
                response.status_code,
                url,
                attempt + 1,
                self._max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("Unreachable")

    def _load_ticker_cik_map(self, client: httpx.Client) -> dict[str, str]:
        payload: Any = None
        cache_file = (
            self.cache_dir / "company_tickers.json" if self.cache_dir is not None else None
        )
        if cache_file is not None and cache_file.exists():
            try:
                age_sec = time.time() - cache_file.stat().st_mtime
                if age_sec <= _TICKERS_CACHE_MAX_AGE_SECONDS:
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload is None:
            response = self._get(client, self.tickers_url)
            payload = response.json()
            if cache_file is not None and isinstance(payload, (dict, list)):
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = cache_file.with_suffix(".json.tmp")
                    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
                    tmp_path.replace(cache_file)
                except OSError as exc:
                    logger.warning("Failed to cache SEC company_tickers.json: %s", exc)

        entries: list[Any]
        if isinstance(payload, dict):
            entries = list(payload.values())
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []

        cik_map: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker") or "").strip().upper()
            raw_cik = entry.get("cik_str")
            if not ticker or raw_cik is None:
                continue
            try:
                cik_int = int(raw_cik)
            except (TypeError, ValueError):
                continue
            cik_str = f"{cik_int:010d}"
            cik_map[ticker] = cik_str
        return cik_map

    @staticmethod
    def _resolve_cik(symbol: str, cik_map: dict[str, str]) -> str | None:
        sym = symbol.strip().upper()
        candidates: list[str] = [sym]
        for alias in SYMBOL_QUERY_ALIASES.get(sym, ()):
            if alias not in candidates:
                candidates.append(alias)
        canonical = SYMBOL_CANONICAL_MAP.get(sym)
        if canonical and canonical not in candidates:
            candidates.append(canonical)
        for cand in list(candidates):
            if "." in cand:
                candidates.append(cand.replace(".", "-"))
            if "-" in cand:
                candidates.append(cand.replace("-", "."))
        for cand in candidates:
            if cand in cik_map:
                return cik_map[cand]
        return None

    @staticmethod
    def _archive_overlaps_window(
        file_meta: dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> bool:
        raw_from = str(file_meta.get("filingFrom") or "").strip()
        raw_to = str(file_meta.get("filingTo") or "").strip()
        try:
            f_from = date.fromisoformat(raw_from[:10]) if raw_from else None
            f_to = date.fromisoformat(raw_to[:10]) if raw_to else None
        except ValueError:
            return True
        if f_from is not None and f_from > end_date + timedelta(days=5):
            return False
        return not (f_to is not None and f_to < start_date - timedelta(days=5))

    @staticmethod
    def _extract_candidate_rows(
        block: Any,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[tuple[str, Literal["4", "8-K"], datetime, str, list[str]]]:
        if not isinstance(block, dict):
            return []
        accessions = block.get("accessionNumber") or []
        forms = block.get("form") or []
        acc_dts = block.get("acceptanceDateTime") or []
        filing_dates = block.get("filingDate") or []
        primary_docs = block.get("primaryDocument") or []
        items_col = block.get("items") or []

        n = len(accessions)
        rows: list[tuple[str, Literal["4", "8-K"], datetime, str, list[str]]] = []
        for idx in range(n):
            raw_form = str(forms[idx] if idx < len(forms) else "").strip().upper()
            if raw_form not in {"4", "8-K"}:
                continue
            form_type: Literal["4", "8-K"] = "4" if raw_form == "4" else "8-K"

            acc_num = str(accessions[idx] or "").strip()
            if not acc_num:
                continue
            raw_acc_dt = acc_dts[idx] if idx < len(acc_dts) else None
            raw_fd = filing_dates[idx] if idx < len(filing_dates) else None
            accepted_at = _parse_sec_acceptance_dt(raw_acc_dt, raw_fd)
            if accepted_at is None:
                continue
            if accepted_at < start_utc or accepted_at > end_utc:
                continue

            primary_doc = str(primary_docs[idx] if idx < len(primary_docs) else "").strip()
            raw_items = items_col[idx] if idx < len(items_col) else []
            items = _normalize_items(raw_items) if form_type == "8-K" else []
            rows.append((acc_num, form_type, accepted_at, primary_doc, items))
        return rows

    def fetch_filings(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[SecFilingEvent]:
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise ValueError("end must be >= start")

        requested_symbols = list(
            dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip())
        )
        if not requested_symbols:
            return []

        events_by_key: dict[tuple[str, str], SecFilingEvent] = {}
        own_client = self._external_client is None
        client = self._external_client or httpx.Client(timeout=self._timeout)

        try:
            cik_map = self._load_ticker_cik_map(client)
            start_d = start_utc.date()
            end_d = end_utc.date()

            for sym in requested_symbols:
                cik = self._resolve_cik(sym, cik_map)
                if cik is None:
                    logger.debug("Skipping SEC EDGAR lookup for %s (no CIK match)", sym)
                    continue

                sub_url = f"{self.submissions_base_url}/CIK{cik}.json"
                try:
                    sub_resp = self._get(client, sub_url)
                    sub_payload = sub_resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("Failed to fetch SEC submissions for %s (%s): %s", sym, cik, exc)
                    continue

                if not isinstance(sub_payload, dict):
                    continue
                filings_obj = sub_payload.get("filings")
                if not isinstance(filings_obj, dict):
                    continue

                candidates = self._extract_candidate_rows(
                    filings_obj.get("recent"), start_utc, end_utc
                )

                archive_files = filings_obj.get("files") or []
                if isinstance(archive_files, list):
                    for file_meta in archive_files:
                        if not isinstance(file_meta, dict):
                            continue
                        fname = str(file_meta.get("name") or "").strip()
                        if not fname or not self._archive_overlaps_window(file_meta, start_d, end_d):
                            continue
                        archive_url = f"{self.submissions_base_url}/{fname.lstrip('/')}"
                        try:
                            arch_resp = self._get(client, archive_url)
                            arch_payload = arch_resp.json()
                        except (httpx.HTTPError, ValueError) as exc:
                            logger.warning(
                                "Failed to fetch SEC archive %s for %s: %s", fname, sym, exc
                            )
                            continue
                        candidates.extend(
                            self._extract_candidate_rows(arch_payload, start_utc, end_utc)
                        )

                deduped_8k: dict[str, tuple[datetime, list[str]]] = {}
                form4_candidates: dict[str, tuple[datetime, str]] = {}

                for acc_num, form_type, accepted_at, primary_doc, items in candidates:
                    if form_type == "8-K":
                        deduped_8k[acc_num] = (accepted_at, items)
                    else:
                        form4_candidates[acc_num] = (accepted_at, primary_doc)

                for acc_num, (accepted_at, items) in deduped_8k.items():
                    events_by_key[(acc_num, sym)] = SecFilingEvent(
                        accession_number=acc_num,
                        symbol=sym,
                        cik=cik,
                        form_type="8-K",
                        accepted_at=accepted_at,
                        items=items,
                    )

                if self.max_form4_per_symbol == 0:
                    continue

                sorted_form4 = sorted(
                    form4_candidates.items(),
                    key=lambda kv: (kv[1][0], kv[0]),
                    reverse=True,
                )[: self.max_form4_per_symbol]

                cik_int_str = str(int(cik))
                for acc_num, (accepted_at, primary_doc) in sorted_form4:
                    raw_doc = primary_doc.strip().rsplit("/", 1)[-1]
                    if not raw_doc:
                        continue
                    acc_nodash = acc_num.replace("-", "")
                    xml_url = f"{self.archives_base_url}/{cik_int_str}/{acc_nodash}/{raw_doc}"
                    try:
                        xml_resp = self._get(client, xml_url)
                        net_buy_usd, buyer_count, seller_count = _parse_form4_xml(xml_resp.text)
                    except (httpx.HTTPError, ET.ParseError) as exc:
                        logger.warning(
                            "Skipping malformed/unreadable Form 4 %s for %s: %s",
                            acc_num,
                            sym,
                            exc,
                        )
                        continue

                    events_by_key[(acc_num, sym)] = SecFilingEvent(
                        accession_number=acc_num,
                        symbol=sym,
                        cik=cik,
                        form_type="4",
                        accepted_at=accepted_at,
                        items=[],
                        net_insider_buy_usd=net_buy_usd,
                        insider_buyer_count=buyer_count,
                        insider_seller_count=seller_count,
                    )
        finally:
            if own_client:
                client.close()

        return sorted(
            events_by_key.values(),
            key=lambda ev: (ev.symbol, ev.accepted_at, ev.accession_number),
        )
