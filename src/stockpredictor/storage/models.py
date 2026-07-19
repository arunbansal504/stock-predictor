"""SQLAlchemy ORM models for the relational reference/app store (§13).

Phase 1 scope: `securities`, `corporate_actions`, `run_metadata`, plus the ML
Review Board governance tables `published_predictions`/`validation_results`
(weekly-published recommendations and their resolved outcomes — see
reporting/publish.py and reporting/validation.py). Tables for `users`,
`watchlists`, `user_portfolios`, `alerts`, `audit_log` are Phase 2 (§27) and
intentionally not modeled yet — adding them now, unused, would violate the
"earn its place" discipline from the architecture doc's Truth 3.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Security(Base):
    """The tradable-universe master list (§13). `symbol` is the exchange
    ticker without any data-provider suffix (e.g. "RELIANCE", not
    "RELIANCE.NS") — connectors are responsible for provider-specific
    formatting (see connectors/prices_yfinance.py)."""

    __tablename__ = "securities"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    exchange: Mapped[str] = mapped_column(String(8), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    added_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    corporate_actions: Mapped[list["CorporateAction"]] = relationship(
        back_populates="security", cascade="all, delete-orphan"
    )


class CorporateAction(Base):
    """Splits/bonuses/dividends/buybacks, PIT-stamped (§5, §13).

    `ex_date` is the event date (the corporate action's effective date);
    `knowable_date` is when it was publicly announced/knowable — usually
    <= ex_date, but modeled explicitly rather than assumed, per the PIT
    discipline in common/pit.py.
    """

    __tablename__ = "corporate_actions"
    __table_args__ = (
        UniqueConstraint("symbol", "action_type", "ex_date", name="uq_corp_action_natural_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("securities.symbol"), nullable=False)
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)  # split|bonus|dividend|buyback
    ex_date: Mapped[dt.date] = mapped_column(nullable=False)
    knowable_date: Mapped[dt.date] = mapped_column(nullable=False)
    ratio: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)
    value: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)

    security: Mapped[Security] = relationship(back_populates="corporate_actions")


class RunMetadata(Base):
    """One row per orchestration DAG stage per run (§13, §22, §23) — the audit
    trail that makes a nightly run's outcome (and any failure) inspectable
    without digging through logs."""

    __tablename__ = "run_metadata"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # running|success|failed
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rows_processed: Mapped[int | None] = mapped_column(nullable=True)
    detail: Mapped[str | None] = mapped_column(String(2048), nullable=True)


class PublishedPrediction(Base):
    """One row per symbol in a weekly-published Top-N recommendation set
    (reporting/publish.py, ML Review Board spec Part 1). Deliberately
    separate from the Gold `predictions`/`rankings` Parquet domains (which
    accumulate on every *nightly* run and are the model's rolling internal
    state) -- this table is the append-only, human-facing record of what was
    actually officially recommended on a given publish date, enriched with
    fields (buy_price, confidence, feature snapshots, model/commit
    provenance) that the internal scoring tables don't carry. Rows are never
    updated or deleted after insert; see publish.py's never-overwrite
    discipline.
    """

    __tablename__ = "published_predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    prediction_date: Mapped[dt.date] = mapped_column(nullable=False, index=True)
    prediction_horizon: Mapped[str] = mapped_column(String(8), nullable=False)
    stock_symbol: Mapped[str] = mapped_column(ForeignKey("securities.symbol"), nullable=False)
    # buy_price stays Numeric (a currency-like quoted price); the rest are
    # statistical scores/ratios, not exact-decimal values -- Float so a
    # plain ORM read gives a native Python float instead of decimal.Decimal
    # (Numeric's Python-side type), which every reporting/analytics
    # consumer downstream otherwise has to remember to cast.
    buy_price: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    prediction_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(nullable=False)
    # meta_score -- the model's raw, pre-calibration signal. Same meaning
    # "relative strength" already has in the Streamlit UI/USER_GUIDE.md
    # glossary (apps/streamlit_app/app.py renames meta_score to
    # "relative_strength" for display) -- deliberately kept as one
    # consistent definition rather than a second, different one under the
    # same name.
    relative_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    disagreement: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON-serialized (same convention as explain/registry.py's _JSON_COLUMNS)
    # rather than nested Parquet-unsafe structures.
    technical_features: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment_features: Mapped[str] = mapped_column(Text, nullable=False)
    feature_vector: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    git_commit_hash: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    validation: Mapped["ValidationResult | None"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class ValidationResult(Base):
    """Outcome of a published prediction once its horizon has resolved
    (reporting/validation.py, ML Review Board spec Part 2). One row per
    `PublishedPrediction`, written once and never updated -- a resolved
    outcome is a historical fact, not a value to overwrite on a later
    validation run."""

    __tablename__ = "validation_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("published_predictions.prediction_id"), unique=True, nullable=False, index=True
    )
    actual_return: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=False)
    alpha: Mapped[float] = mapped_column(Float, nullable=False)
    hit_or_miss: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Nullable: short/edge-case holding windows (e.g. a 5d horizon spanning a
    # holiday cluster) can have too few daily observations for these to be
    # defined -- see backtest/metrics.py's own NaN-on-insufficient-data
    # convention, which this mirrors rather than silently coercing to 0.
    maximum_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    maximum_gain: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    information_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    prediction: Mapped[PublishedPrediction] = relationship(back_populates="validation")
