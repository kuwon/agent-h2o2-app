# db/tables/pension_models.py

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
    customer_name: Mapped[str] = mapped_column(String)
    brth_dt: Mapped[str] = mapped_column(String(8), comment='생년월일: Birth Date (YYYYMMDD)')
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
        PrimaryKeyConstraint("account_id", "prd_type_cd", name="kis_accounts_pkey"),
        {'schema': 'ai'}
    )

    account_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, nullable=False, index=True
    )
    customer_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    acnt_type: Mapped[str] = mapped_column(String, comment='계좌유형: Account Type')
    prd_type_cd: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False, comment='상품코드: product type code')
    acnt_bgn_dt: Mapped[str] = mapped_column(String(8), comment='계좌 개설 일시: Account Begin Date (YYYYMMDD)')
    acnt_evlu_amt: Mapped[int] = mapped_column(BigInteger, comment='계좌 평가액: Account Evaluation Amount')
    rcve_odyr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment='수령연차: Receive Order of Year')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=text("now()")
    )


class DefinedContributionContractTable(Base):
    """Table for storing contract info for DC"""

    __tablename__ = "kis_dc_contract"
    __table_args__ = ({'schema': 'ai'},)

    ctrt_no: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, nullable=False, index=True
    )
    odtp_name: Mapped[str] = mapped_column(String, comment='근무처명: On Duty Place Name')
    etco_dt: Mapped[str] = mapped_column(String(8), comment='입사일자: Entering a Company Date')
    midl_excc_dt: Mapped[str] = mapped_column(String(8), comment='중간정산일자: Middle Exact Calculation Date')
    sst_join_dt: Mapped[str] = mapped_column(String(8), comment='제도가입일자: System join Date')
    almt_pymt_prca: Mapped[str] = mapped_column(BigInteger, comment='부담금납입원금: Allotment Payment Principal')
    utlz_pfls_amt: Mapped[int] = mapped_column(BigInteger, comment='운용손익금액: Utilization Profitloss Amount')
    evlu_acca_smtl_amt: Mapped[int] = mapped_column(BigInteger, comment='평가적립금합계금액: Evaluation Accumulated Fund Sum')

