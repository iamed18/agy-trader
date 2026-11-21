from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from agy_trader.core.models import NewsArticle, ensure_utc


@dataclass(frozen=True, slots=True)
class SentimentPrediction:
    score: float  # Continuous polarity in [-1.0, 1.0]
    label: str  # "positive" | "negative" | "neutral"
    confidence: float  # Confidence in [0.0, 1.0]
    scorer_model: str


class SentimentScorer(ABC):
    """Pluggable quantitative sentiment scorer interface."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def score_texts(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        """Score a batch of financial headlines/summaries."""

    def score_articles(self, articles: Sequence[NewsArticle]) -> list[SentimentPrediction]:
        texts = [
            f"{a.headline}. {a.summary}".strip() if a.summary.strip() else a.headline.strip()
            for a in articles
        ]
        return self.score_texts(texts)


# Curated financial polarity lexicon (Loughran-McDonald / earnings / analyst action inspired).
_POSITIVE_PHRASES: dict[str, float] = {
    "beats estimates": 1.6,
    "beat estimates": 1.6,
    "beats expectations": 1.6,
    "raises guidance": 1.8,
    "raised guidance": 1.8,
    "raises outlook": 1.6,
    "price target raised": 1.4,
    "record revenue": 1.5,
    "record earnings": 1.5,
    "strong demand": 1.3,
    "share buyback": 1.1,
    "stock buyback": 1.1,
    "dividend hike": 1.2,
    "dividend increase": 1.2,
    "fda approval": 1.8,
    "upgraded to buy": 1.6,
    "upgraded to outperform": 1.5,
    "upgraded to overweight": 1.4,
}

_NEGATIVE_PHRASES: dict[str, float] = {
    "misses estimates": 1.7,
    "missed estimates": 1.7,
    "misses expectations": 1.7,
    "cuts guidance": 1.9,
    "cut guidance": 1.9,
    "lowers guidance": 1.8,
    "lowered guidance": 1.8,
    "lowers outlook": 1.6,
    "weak outlook": 1.6,
    "weak revenue": 1.5,
    "price target cut": 1.4,
    "price target lowered": 1.4,
    "sec investigation": 2.0,
    "accounting probe": 2.0,
    "product recall": 1.6,
    "downgraded to sell": 1.7,
    "downgraded to underperform": 1.5,
    "downgraded to underweight": 1.4,
    "dividend cut": 1.8,
    "layoffs announced": 1.2,
}

_POSITIVE_TOKENS: dict[str, float] = {
    "beat": 0.9,
    "beats": 0.9,
    "surge": 1.0,
    "surges": 1.0,
    "surged": 1.0,
    "soar": 1.0,
    "soars": 1.0,
    "rally": 0.9,
    "rallies": 0.9,
    "upgrade": 1.1,
    "upgraded": 1.1,
    "outperform": 1.0,
    "overweight": 0.8,
    "bullish": 1.0,
    "growth": 0.6,
    "accelerate": 0.8,
    "accelerates": 0.8,
    "accelerating": 0.8,
    "accelerated": 0.8,
    "strong": 0.7,
    "profit": 0.5,
    "profitable": 0.7,
    "record": 0.7,
    "expand": 0.6,
    "expansion": 0.7,
    "exceeds": 1.0,
    "exceeded": 1.0,
    "tops": 0.8,
    "breakout": 0.8,
    "approval": 1.0,
    "approved": 1.0,
    "raises": 0.7,
    "raised": 0.7,
    "momentum": 0.5,
}

_NEGATIVE_TOKENS: dict[str, float] = {
    "miss": 1.0,
    "misses": 1.0,
    "missed": 1.0,
    "plunge": 1.2,
    "plunges": 1.2,
    "plunged": 1.2,
    "slump": 1.0,
    "slumps": 1.0,
    "tumble": 1.0,
    "tumbles": 1.0,
    "drop": 0.7,
    "drops": 0.7,
    "decline": 0.8,
    "declines": 0.8,
    "downgrade": 1.2,
    "downgraded": 1.2,
    "underperform": 1.1,
    "underweight": 0.9,
    "bearish": 1.0,
    "weak": 0.8,
    "weakness": 0.9,
    "loss": 0.8,
    "losses": 0.9,
    "cut": 0.7,
    "cuts": 0.7,
    "slashed": 1.1,
    "warns": 1.1,
    "warning": 1.1,
    "lawsuit": 1.1,
    "probe": 1.0,
    "investigation": 1.1,
    "recall": 1.1,
    "default": 1.5,
    "bankruptcy": 1.8,
    "fraud": 1.8,
    "layoffs": 0.9,
    "headwinds": 0.8,
    "slowdown": 0.9,
    "shortfall": 1.0,
}

_NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "no", "never", "without", "fails", "failed", "failing", "despite"}
)

_TOKEN_RE = re.compile(r"[a-z]+")


class LexiconSentimentScorer(SentimentScorer):
    """Deterministic zero-ML financial lexicon sentiment scorer."""

    @property
    def model_name(self) -> str:
        return "lexicon-v1"

    def score_texts(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        results: list[SentimentPrediction] = []
        for raw_text in texts:
            lowered = raw_text.lower()
            pos_weight = 0.0
            neg_weight = 0.0

            tokens = _TOKEN_RE.findall(lowered)
            normalized_space = " ".join(tokens)

            for phrase, weight in _POSITIVE_PHRASES.items():
                match = re.search(rf"(?:^|\b)(\w+\s+)?{re.escape(phrase)}\b", normalized_space)
                if match:
                    prev_word = (match.group(1) or "").strip()
                    if prev_word in _NEGATION_WORDS:
                        neg_weight += weight * 0.8
                    else:
                        pos_weight += weight

            for phrase, weight in _NEGATIVE_PHRASES.items():
                match = re.search(rf"(?:^|\b)(\w+\s+)?{re.escape(phrase)}\b", normalized_space)
                if match:
                    prev_word = (match.group(1) or "").strip()
                    if prev_word in _NEGATION_WORDS:
                        pos_weight += weight * 0.6
                    else:
                        neg_weight += weight

            for idx, tok in enumerate(tokens):
                negated = idx > 0 and tokens[idx - 1] in _NEGATION_WORDS
                if tok in _POSITIVE_TOKENS:
                    w = _POSITIVE_TOKENS[tok]
                    if negated:
                        neg_weight += w * 0.8
                    else:
                        pos_weight += w
                elif tok in _NEGATIVE_TOKENS:
                    w = _NEGATIVE_TOKENS[tok]
                    if negated:
                        pos_weight += w * 0.6
                    else:
                        neg_weight += w

            total = pos_weight + neg_weight
            if total <= 1e-9:
                results.append(
                    SentimentPrediction(
                        score=0.0,
                        label="neutral",
                        confidence=0.35,
                        scorer_model=self.model_name,
                    )
                )
                continue

            net = pos_weight - neg_weight
            # Smooth bounded polarity in [-1.0, 1.0]
            score = max(-1.0, min(1.0, math.tanh(net / 2.2)))
            dominance = abs(net) / total
            confidence = max(0.35, min(0.95, 0.40 + 0.45 * dominance * math.tanh(total / 2.0)))

            if score > 0.12:
                label = "positive"
            elif score < -0.12:
                label = "negative"
            else:
                label = "neutral"

            results.append(
                SentimentPrediction(
                    score=round(score, 6),
                    label=label,
                    confidence=round(confidence, 6),
                    scorer_model=self.model_name,
                )
            )
        return results


class FinBERTScorer(SentimentScorer):
    """Local ProsusAI/finbert sentiment scorer (CPU or GPU auto-detect, lazy-loaded)."""

    def __init__(self, model_id: str = "ProsusAI/finbert", batch_size: int = 16) -> None:
        from typing import Any

        self._model_id = model_id
        self._batch_size = max(1, batch_size)
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._label_map: dict[int, str] = {}

    @property
    def model_name(self) -> str:
        return self._model_id

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT requires optional NLP dependencies. "
                "Install with: uv sync --extra nlp"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()
        self._label_map = {
            int(k): str(v).lower() for k, v in self._model.config.id2label.items()
        }

    def score_texts(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        if not texts:
            return []
        self._ensure_loaded()
        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        predictions: list[SentimentPrediction] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for row in probs:
                prob_by_label = {self._label_map[i]: float(p) for i, p in enumerate(row)}
                p_pos = prob_by_label.get("positive", 0.0)
                p_neg = prob_by_label.get("negative", 0.0)
                score = max(-1.0, min(1.0, p_pos - p_neg))
                best_label = max(prob_by_label, key=lambda lbl: prob_by_label[lbl])
                confidence = max(0.0, min(1.0, prob_by_label[best_label]))
                predictions.append(
                    SentimentPrediction(
                        score=round(score, 6),
                        label=best_label,
                        confidence=round(confidence, 6),
                        scorer_model=self.model_name,
                    )
                )
        return predictions


def create_sentiment_scorer(backend: str = "lexicon") -> SentimentScorer:
    normalized = backend.strip().lower()
    if normalized == "lexicon":
        return LexiconSentimentScorer()
    if normalized == "finbert":
        return FinBERTScorer()
    raise ValueError(f"Unsupported NLP backend: {backend!r} (expected 'lexicon' or 'finbert')")


@dataclass(frozen=True, slots=True)
class AggregatedSentiment:
    news_score: float  # Decayed polarity amplified gently by positive velocity surprise
    raw_polarity: float  # Pure confidence-weighted decayed polarity in [-1.0, 1.0]
    velocity_z: float  # 24h headline count z-score vs prior 29d daily baseline
    article_count_window: int


class SentimentAggregator:
    """Strictly point-in-time news sentiment & velocity aggregator (zero lookahead)."""

    def __init__(
        self,
        half_life_hours: float = 24.0,
        lookback_hours: float = 72.0,
        baseline_days: int = 30,
    ) -> None:
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be > 0")
        if lookback_hours < half_life_hours:
            raise ValueError("lookback_hours must be >= half_life_hours")
        if baseline_days < 2:
            raise ValueError("baseline_days must be >= 2")
        self.half_life_hours = half_life_hours
        self.lookback_hours = lookback_hours
        self.baseline_days = baseline_days

    def aggregate_for_symbol(
        self,
        articles: Sequence[NewsArticle],
        as_of: datetime,
    ) -> AggregatedSentiment:
        """Aggregate news strictly with published_at <= as_of (UTC)."""
        as_of_utc = ensure_utc(as_of)
        baseline_start = as_of_utc - timedelta(days=self.baseline_days)
        window_start = as_of_utc - timedelta(hours=self.lookback_hours)
        day_start = as_of_utc - timedelta(hours=24)

        # Deduplicate by headline_hash while enforcing strict published_at <= as_of_utc.
        deduped: dict[str, NewsArticle] = {}
        for art in articles:
            pub = ensure_utc(art.published_at)
            if pub > as_of_utc or pub < baseline_start:
                continue
            if art.sentiment_score is None:
                continue
            existing = deduped.get(art.headline_hash)
            if existing is None or pub < existing.published_at:
                deduped[art.headline_hash] = art

        if not deduped:
            return AggregatedSentiment(
                news_score=0.0,
                raw_polarity=0.0,
                velocity_z=0.0,
                article_count_window=0,
            )

        ln2 = math.log(2.0)
        weighted_sum = 0.0
        weight_total = 0.0
        count_window = 0
        count_24h = 0

        # Prior (baseline_days - 1) daily buckets strictly before (as_of - 24h) so
        # today's burst never contaminates its own baseline distribution.
        prior_buckets = [0] * (self.baseline_days - 1)

        for art in deduped.values():
            pub = art.published_at
            if pub >= window_start:
                age_hours = max(0.0, (as_of_utc - pub).total_seconds() / 3600.0)
                decay = math.exp(-ln2 * age_hours / self.half_life_hours)
                conf = art.confidence if art.confidence is not None else 0.5
                weight = decay * max(conf, 0.1)
                assert art.sentiment_score is not None
                weighted_sum += weight * art.sentiment_score
                weight_total += weight
                count_window += 1

            if pub > day_start:
                count_24h += 1
            else:
                # Bucket index 0 is (as_of - 48h, as_of - 24h], etc.
                age_from_day_start_sec = (day_start - pub).total_seconds()
                bucket_idx = int(age_from_day_start_sec // 86400.0)
                if 0 <= bucket_idx < len(prior_buckets):
                    prior_buckets[bucket_idx] += 1

        if count_window == 0 or weight_total <= 1e-12:
            return AggregatedSentiment(
                news_score=0.0,
                raw_polarity=0.0,
                velocity_z=0.0,
                article_count_window=0,
            )

        raw_polarity = max(-1.0, min(1.0, weighted_sum / weight_total))

        n_buckets = len(prior_buckets)
        mean_daily = sum(prior_buckets) / n_buckets
        var_daily = sum((c - mean_daily) ** 2 for c in prior_buckets) / max(n_buckets - 1, 1)
        # Floor std at 0.5 articles/day so quiet tickers don't explode on a single article.
        std_daily = max(math.sqrt(var_daily), 0.5)
        velocity_z = max(-3.0, min(5.0, (count_24h - mean_daily) / std_daily))

        # Positive volume bursts amplify polarity up to 1.5x without ever flipping sign.
        velocity_amp = 1.0 + 0.5 * math.tanh(max(velocity_z, 0.0) / 2.0)
        news_score = max(-1.5, min(1.5, raw_polarity * velocity_amp))

        return AggregatedSentiment(
            news_score=round(news_score, 6),
            raw_polarity=round(raw_polarity, 6),
            velocity_z=round(velocity_z, 6),
            article_count_window=count_window,
        )
