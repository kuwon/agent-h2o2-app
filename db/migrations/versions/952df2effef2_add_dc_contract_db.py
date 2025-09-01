"""add dc contract db (safe: skip if exists, align PK)

Revision ID: 952df2effef2
Revises: 27728246f29e
Create Date: 2025-09-01 13:46:26.533321
"""
from alembic import op
import sqlalchemy as sa

revision = "952df2effef2"
down_revision = "27728246f29e"
branch_labels = None
depends_on = None

SCHEMA = "ai"


def _has_table(bind, name: str) -> bool:
    insp = sa.inspect(bind)
    return insp.has_table(name, schema=SCHEMA)


def _get_pk_cols(bind, table: str):
    insp = sa.inspect(bind)
    # ✅ 테이블명은 그대로, 스키마는 인자로 분리
    pk = insp.get_pk_constraint(table, schema=SCHEMA)
    return set(pk.get("constrained_columns", [])), pk.get("name")


def _index_exists(bind, schema: str, table: str, index_name: str) -> bool:
    # pg_indexes 조회로 인덱스 존재 확인
    res = bind.execute(
        sa.text(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = :schema
              AND tablename  = :table
              AND indexname  = :index
            """
        ),
        {"schema": schema, "table": table, "index": index_name},
    ).first()
    return res is not None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # --- kis_accounts ---
    if not _has_table(bind, "kis_accounts"):
        op.create_table(
            "kis_accounts",
            sa.Column("account_id", sa.BigInteger(), autoincrement=False, nullable=False),
            sa.Column("customer_id", sa.BigInteger(), nullable=False),
            sa.Column("acnt_type", sa.String(), nullable=False, comment="계좌유형: Account Type"),
            sa.Column("prd_type_cd", sa.Integer(), nullable=False, comment="상품코드: product type code"),
            sa.Column("acnt_bgn_dt", sa.String(length=8), nullable=False, comment="계좌 개설 일시: Account Begin Date (YYYYMMDD)"),
            sa.Column("acnt_evlu_amt", sa.BigInteger(), nullable=False, comment="계좌 평가액: Account Evaluation Amount"),
            sa.Column("rcve_odyr", sa.Integer(), nullable=True, comment="수령연차: Receive Order of Year"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("account_id", "prd_type_cd", name="kis_accounts_pkey"),
            schema=SCHEMA,
        )
    else:
        # 이미 존재하면 PK 정합 확인
        pk_cols, pk_name = _get_pk_cols(bind, "kis_accounts")
        desired = {"account_id", "prd_type_cd"}
        if pk_cols != desired:
            op.drop_constraint(pk_name, "kis_accounts", type_="primary", schema=SCHEMA)
            op.create_primary_key("kis_accounts_pkey", "kis_accounts", ["account_id", "prd_type_cd"], schema=SCHEMA)

    # 인덱스 생성 (존재 시 스킵)
    if not _index_exists(bind, SCHEMA, "kis_accounts", "ix_ai_kis_accounts_account_id"):
        op.create_index("ix_ai_kis_accounts_account_id", "kis_accounts", ["account_id"], unique=False, schema=SCHEMA)
    if not _index_exists(bind, SCHEMA, "kis_accounts", "ix_ai_kis_accounts_customer_id"):
        op.create_index("ix_ai_kis_accounts_customer_id", "kis_accounts", ["customer_id"], unique=False, schema=SCHEMA)

    # --- kis_customers ---
    if not _has_table(bind, "kis_customers"):
        op.create_table(
            "kis_customers",
            sa.Column("customer_id", sa.BigInteger(), autoincrement=False, nullable=False),
            sa.Column("customer_name", sa.String(), nullable=False),
            sa.Column("brth_dt", sa.String(length=8), nullable=False, comment="생년월일: Birth Date (YYYYMMDD)"),
            sa.Column("age_band", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("customer_id"),
            schema=SCHEMA,
        )

    if not _index_exists(bind, SCHEMA, "kis_customers", "ix_ai_kis_customers_customer_id"):
        op.create_index("ix_ai_kis_customers_customer_id", "kis_customers", ["customer_id"], unique=False, schema=SCHEMA)

    # --- kis_dc_contract ---
    if not _has_table(bind, "kis_dc_contract"):
        op.create_table(
            "kis_dc_contract",
            sa.Column("ctrt_no", sa.BigInteger(), autoincrement=False, nullable=False),
            sa.Column("odtp_name", sa.String(), nullable=False, comment="근무처명: On Duty Place Name"),
            sa.Column("etco_dt", sa.String(length=8), nullable=False, comment="입사일자: Entering a Company Date"),
            sa.Column("midl_excc_dt", sa.String(length=8), nullable=False, comment="중간정산일자: Middle Exact Calculation Date"),
            sa.Column("almt_pymt_prca", sa.String(length=8), nullable=False, comment="부담금납입원금: Allotment Payment Principal"),
            sa.Column("utlz_pfls_amt", sa.BigInteger(), nullable=False, comment="운용손익금액: Utilization Profitloss Amount"),
            sa.Column("evlu_acca_smtl_amt", sa.BigInteger(), nullable=False, comment="평가적립금합계금액: Evaluation Accumulated Fund Sum"),
            sa.PrimaryKeyConstraint("ctrt_no"),
            schema=SCHEMA,
        )

    if not _index_exists(bind, SCHEMA, "kis_dc_contract", "ix_ai_kis_dc_contract_ctrt_no"):
        op.create_index("ix_ai_kis_dc_contract_ctrt_no", "kis_dc_contract", ["ctrt_no"], unique=False, schema=SCHEMA)


def downgrade() -> None:
    bind = op.get_bind()

    if _index_exists(bind, SCHEMA, "kis_dc_contract", "ix_ai_kis_dc_contract_ctrt_no"):
        op.drop_index("ix_ai_kis_dc_contract_ctrt_no", table_name="kis_dc_contract", schema=SCHEMA)
    if _has_table(bind, "kis_dc_contract"):
        op.drop_table("kis_dc_contract", schema=SCHEMA)

    if _index_exists(bind, SCHEMA, "kis_customers", "ix_ai_kis_customers_customer_id"):
        op.drop_index("ix_ai_kis_customers_customer_id", table_name="kis_customers", schema=SCHEMA)
    if _has_table(bind, "kis_customers"):
        op.drop_table("kis_customers", schema=SCHEMA)

    if _index_exists(bind, SCHEMA, "kis_accounts", "ix_ai_kis_accounts_customer_id"):
        op.drop_index("ix_ai_kis_accounts_customer_id", table_name="kis_accounts", schema=SCHEMA)
    if _index_exists(bind, SCHEMA, "kis_accounts", "ix_ai_kis_accounts_account_id"):
        op.drop_index("ix_ai_kis_accounts_account_id", table_name="kis_accounts", schema=SCHEMA)
    if _has_table(bind, "kis_accounts"):
        op.drop_table("kis_accounts", schema=SCHEMA)
