# db/tables/simulation_models.py

from datetime import datetime
from typing import Optional

from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.expression import text
from sqlalchemy.types import BigInteger, DateTime, String, Integer

from db.tables.base import Base


class CustomersTable(Base):
    """Table for storing customers data (from '고객통합기본')."""

    __tablename__ = "kis_customers"
    __table_args__ = {'schema': 'ai'}

    # CSNO
    customer_id: Mapped[Optional[str]] = mapped_column(
        String(9), primary_key=True, autoincrement=False, nullable=False, index=True,
        comment="고객번호: Customer ID"
    )
    # CUST_NAME
    customer_name: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, comment="고객명: Customer Name"
    )
    # BRDT (YYYYMMDD)
    brth_dt: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True, comment="생년월일(YYYYMMDD): Birth Date"
    )
    # TOT_ASST_AMT
    tot_asst_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="총자산금액: Total Asset Amount"
    )
    # CUST_IVST_ICLN_GRAD_CD
    cust_ivst_icln_grad_cd: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, comment="고객투자성향등급: Investment Risk Grade"
    )

    # created/updated
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=text("now()")
    )


class AccountsTable(Base):
    """Table for storing accounts data (from '계좌정보')."""

    __tablename__ = "kis_accounts"
    __table_args__ = (
        # 복합 PK 유지 (account_id + prd_type_cd)
        PrimaryKeyConstraint("account_id", "prd_type_cd", name="kis_accounts_pkey"),
        {'schema': 'ai'}
    )

    # CANO
    account_id: Mapped[Optional[str]] = mapped_column(
        String(8), primary_key=True, autoincrement=False, nullable=False, index=True,
        comment="계좌번호: Account ID"
    )
    # CSNO
    customer_id: Mapped[Optional[str]] = mapped_column(
        String(9), nullable=False, index=True, comment="고객번호: Customer ID"
    )

    # ACNT_TYPE
    acnt_type: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, comment="계좌유형: Account Type"
    )

    # ACNT_PRDT_CD  ※ 중요: 문자열 코드로 보존 (예: '01')
    prd_type_cd: Mapped[str] = mapped_column(
        String(4), primary_key=True, nullable=False, comment="상품코드(문자): Product Type Code"
    )

    # ACNT_OPEN_DT (YYYYMMDD)
    acnt_bgn_dt: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True, comment="계좌개설일자(YYYYMMDD): Account Open Date"
    )

    # 추가 날짜 컬럼들 (모두 시트 기준 문자열 YYYYMMDD)
    expd_dt: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True, comment="만기일자(YYYYMMDD): Expiration Date"
    )
    etco_dt: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True, comment="입사일자(YYYYMMDD): Entering Company Date"
    )
    rtmt_dt: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True, comment="퇴직일자(YYYYMMDD): Retirement Date"
    )
    midl_excc_dt: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True, comment="중간정산일자(YYYYMMDD): Mid Settlement Date"
    )

    # 금액/수치 컬럼들 (정수 위주 → BigInteger, 필요시 Numeric으로 조정 가능)
    acnt_evlu_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="계좌평가액: Account Evaluation Amount (EVLU_AMT)"
    )
    copt_year_pymt_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="회사부담금_연간납입액: Company Annual Payment Amount (COPT_YEAR_PYMT_AMT)"
    )
    other_txtn_ecls_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="기타과세제외금액: Other Tax-excluded Amount (OTHER_TXTN_ECLS_AMT)"
    )
    rtmt_incm_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="퇴직소득금액: Retirement Income Amount (RTMT_INCM_AMT)"
    )
    icdd_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="이자/배당금액: Interest/Dividend Amount (ICDD_AMT)"
    )
    user_almt_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="사용자부담금: User Allotment Amount (USER_ALMT_AMT)"
    )
    sbsr_almt_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="사용자추가납입금: Subscriber Allotment Amount (SBSR_ALMT_AMT)"
    )
    utlz_erng_amt: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="운용손익금액: Utilization Earning Amount (UTLZ_ERNG_AMT)"
    )
    dfr_rtmt_taxa: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="이연퇴직소득세: Deferred Retirement Tax (DFR_RTMT_TAXA)"
    )

    # 생성/수정 시각
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=text("now()")
    )
