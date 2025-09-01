"""alter almt_pymt_prca to bigint

Revision ID: 2b22cc657347
Revises: 2be305e2d46c
Create Date: 2025-09-01 15:43:41.809993

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2b22cc657347'
down_revision = '2be305e2d46c'
branch_labels = None
depends_on = None

SCHEMA = "ai"
TABLE  = "kis_dc_contract"
COL    = "almt_pymt_prca"

def upgrade() -> None:
    # 1) (선택) 일시적으로 NULL 허용으로 완충 (데이터에 NULL/빈문자가 있으면 안전)
    # op.alter_column(TABLE, COL, schema=SCHEMA, existing_type=sa.String(length=8),
    #                 nullable=True)

    # 2) 타입 변경: USING 절로 안전 캐스팅
    op.alter_column(
        TABLE,
        COL,
        schema=SCHEMA,
        existing_type=sa.String(length=8),
        type_=sa.BigInteger(),
        existing_nullable=False,  # 기존이 NOT NULL이면 False, 그렇지 않으면 True로
        postgresql_using=f"NULLIF({COL}, '')::bigint"
    )

    # 3) (선택) 다시 NOT NULL로 조이기 (필요할 때)
    # op.alter_column(TABLE, COL, schema=SCHEMA, nullable=False)

def downgrade() -> None:
    # bigint -> text(8) 롤백 (길이 8 보존은 논리 제약이므로 COMMENT나 CHECK로 관리 가능)
    op.alter_column(
        TABLE,
        COL,
        schema=SCHEMA,
        existing_type=sa.BigInteger(),
        type_=sa.String(length=8),
        postgresql_using=f"{COL}::text"
    )
