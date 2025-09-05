"""recreate kis_customers and kis_accounts

Revision ID: a5ae02a7033b
Revises: 2b22cc657347
Create Date: 2025-09-05 10:31:11.198481

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a5ae02a7033b'
down_revision = '2b22cc657347'
branch_labels = None
depends_on = None


def upgrade():
    schema = "ai"

    # --- 기존 테이블 삭제 ---
    op.drop_table("kis_accounts", schema=schema)
    op.drop_table("kis_customers", schema=schema)

    # --- 새 테이블 생성: kis_customers ---
    op.create_table(
        "kis_customers",
        sa.Column("customer_id", sa.String(9), primary_key=True, nullable=False, comment="고객번호: Customer ID"),
        sa.Column("customer_name", sa.String, nullable=True, comment="고객명: Customer Name"),
        sa.Column("brth_dt", sa.String(8), nullable=True, comment="생년월일(YYYYMMDD): Birth Date"),
        sa.Column("tot_asst_amt", sa.BigInteger, nullable=True, comment="총자산금액: Total Asset Amount"),
        sa.Column("cust_ivst_icln_grad_cd", sa.String(32), nullable=True, comment="고객투자성향등급: Investment Risk Grade"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.text("now()")),
        schema=schema,
    )

    # --- 새 테이블 생성: kis_accounts ---
    op.create_table(
        "kis_accounts",
        sa.Column("account_id", sa.String(8), primary_key=True, nullable=False, comment="계좌번호: Account ID"),
        sa.Column("customer_id", sa.String(9), nullable=False, index=True, comment="고객번호: Customer ID"),
        sa.Column("acnt_type", sa.String, nullable=True, comment="계좌유형: Account Type"),
        sa.Column("prd_type_cd", sa.String(4), primary_key=True, nullable=False, comment="상품코드: Product Type Code"),
        sa.Column("acnt_bgn_dt", sa.String(8), nullable=True, comment="계좌개설일자(YYYYMMDD): Account Open Date"),
        sa.Column("expd_dt", sa.String(8), nullable=True, comment="만기일자(YYYYMMDD): Expiration Date"),
        sa.Column("etco_dt", sa.String(8), nullable=True, comment="입사일자(YYYYMMDD): Entering Company Date"),
        sa.Column("rtmt_dt", sa.String(8), nullable=True, comment="퇴직일자(YYYYMMDD): Retirement Date"),
        sa.Column("midl_excc_dt", sa.String(8), nullable=True, comment="중간정산일자(YYYYMMDD): Mid Settlement Date"),
        sa.Column("acnt_evlu_amt", sa.BigInteger, nullable=True, comment="계좌평가액: Account Evaluation Amount"),
        sa.Column("copt_year_pymt_amt", sa.BigInteger, nullable=True, comment="회사부담금_연간납입액: Company Annual Payment Amount"),
        sa.Column("other_txtn_ecls_amt", sa.BigInteger, nullable=True, comment="기타과세제외금액: Other Tax-excluded Amount"),
        sa.Column("rtmt_incm_amt", sa.BigInteger, nullable=True, comment="퇴직소득금액: Retirement Income Amount"),
        sa.Column("icdd_amt", sa.BigInteger, nullable=True, comment="이자/배당금액: Interest/Dividend Amount"),
        sa.Column("user_almt_amt", sa.BigInteger, nullable=True, comment="사용자부담금: User Allotment Amount"),
        sa.Column("sbsr_almt_amt", sa.BigInteger, nullable=True, comment="사용자추가납입금: Subscriber Allotment Amount"),
        sa.Column("utlz_erng_amt", sa.BigInteger, nullable=True, comment="운용손익금액: Utilization Earning Amount"),
        sa.Column("dfr_rtmt_taxa", sa.BigInteger, nullable=True, comment="이연퇴직소득세: Deferred Retirement Tax"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.text("now()")),
        sa.PrimaryKeyConstraint("account_id", "prd_type_cd", name="kis_accounts_pkey"),
        schema=schema,
    )


def downgrade():
    schema = "ai"
    op.drop_table("kis_accounts", schema=schema)
    op.drop_table("kis_customers", schema=schema)
