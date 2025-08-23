from datetime import datetime
from typing import Optional

from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.expression import text
from sqlalchemy.types import BigInteger, DateTime, String, Integer

from db.tables.base import Base


class CustomersTable(Base):
    """Table for storing customers data."""

    __tablename__ = "kis_customers"
    __table_args__ = {'schema': 'ai'}

    customer_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, nullable=False, index=True
    )
    customer_name: Mapped[String] = mapped_column(String) 
    brth_dt: Mapped[String] = mapped_column(String(8), comment='생년월일: Birth Date (YYYYMMDD)')
    age_band: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=text("now()")
    )

class AccountsTable(Base):
    """Table for storing accounts data."""

    __tablename__ = "kis_accounts"
    __table_args__ = (
        PrimaryKeyConstraint('account_id', 'prd_type_cd'),
        {'schema': 'ai'}
    )
    account_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, nullable=False, index=True
    )
    customer_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    acnt_type : Mapped[String] = mapped_column(String, comment='계좌유형: Account Type')
    prd_type_cd: Mapped[int] = mapped_column(Integer, comment='상품코드: product type code')
    acnt_bgn_dt : Mapped[String] = mapped_column(String(8), comment='계좌 개설 일시: Account Begin Date (YYYYMMDD)')
    acnt_evlu_amt: Mapped[int] = mapped_column(BigInteger, comment='계좌 평가액: Account Evaluation Amount')
    rcve_odyr: Mapped[int] = mapped_column(Integer, nullable=True, comment='수령연차: Receive Order of Year')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=text("now()")
    )