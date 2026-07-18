"""SQLAlchemy ORM models for the relational reference/app store (§13).

Phase 1 scope only: `securities`, `corporate_actions`, `run_metadata`. Tables
for `users`, `watchlists`, `user_portfolios`, `alerts`, `audit_log` are Phase
2 (§27) and intentionally not modeled yet — adding them now, unused, would
violate the "earn its place" discipline from the architecture doc's Truth 3.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, ForeignKey, Numeric, String, UniqueConstraint
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
