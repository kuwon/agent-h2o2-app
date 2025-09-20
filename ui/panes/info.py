from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import asdict, is_dataclass

import traceback

# Charts
import altair as alt

# SQLAlchemy
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# ORM Models (kis_customers, kis_accounts)
from db.tables.simulation_models import CustomersTable, AccountsTable
from db.session import db_url

# UI helpers
from ui.components.aggrid import aggrid_table
from ui.state import PensionContext
from ui.utils import update_ctx

from workspace.utils.db_key_eng_kor import KMAP_ACCOUNTS, KMAP_CUSTOMERS

from datetime import datetime

def ymd_to_iso(s: str) -> str:
    return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")


# --------------------------------------------------------------------------------------
# DB Session (cache)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_session():
    engine = create_engine(db_url, pool_pre_ping=True)
    return sessionmaker(bind=engine)()


# --------------------------------------------------------------------------------------
# Data Access
# --------------------------------------------------------------------------------------
def _log_debug(enabled: bool, title: str, payload):
    if not enabled:
        return
    with st.expander(f"🧪 DEBUG · {title}", expanded=False):
        try:
            st.write("type:", type(payload))
            if isinstance(payload, pd.DataFrame):
                st.write("shape:", payload.shape)
                st.write("columns:", list(payload.columns))
                st.dataframe(payload.head(20))
            elif isinstance(payload, dict):
                st.json(payload)
            else:
                st.write(payload)
        except Exception:
            st.code(traceback.format_exc())


def _load_customers(sess, q: Optional[str] = None, limit: int = 5000) -> pd.DataFrame:
    try:
        stmt = select(
            CustomersTable.customer_id,
            CustomersTable.customer_name,
            CustomersTable.brth_dt,
            CustomersTable.tot_asst_amt,
            CustomersTable.cust_ivst_icln_grad_cd,
        ).limit(limit)

        if q:
            if hasattr(CustomersTable, "customer_name"):
                stmt = stmt.where(CustomersTable.customer_name.ilike(f"%{q}%"))

        rows = sess.execute(stmt).all()
        df = pd.DataFrame(rows, columns=[x for x in KMAP_CUSTOMERS.keys()])
        if "customer_id" in df.columns:
            df["customer_id"] = df["customer_id"].astype(str)
        return df
    except Exception:
        sess.rollback()
        raise


def _load_accounts(sess, customer_id: Optional[str], limit: int = 500) -> pd.DataFrame:
    try:
        stmt = select(
            AccountsTable.account_id,
            AccountsTable.customer_id,
            AccountsTable.acnt_type,
            AccountsTable.prd_type_cd,
            AccountsTable.acnt_bgn_dt,
            AccountsTable.expd_dt,
            AccountsTable.etco_dt,
            AccountsTable.rtmt_dt,
            AccountsTable.midl_excc_dt,
            AccountsTable.acnt_evlu_amt,
            AccountsTable.copt_year_pymt_amt,
            AccountsTable.other_txtn_ecls_amt,
            AccountsTable.rtmt_incm_amt,
            AccountsTable.icdd_amt,
            AccountsTable.user_almt_amt,
            AccountsTable.sbsr_almt_amt,
            AccountsTable.utlz_erng_amt,
            AccountsTable.dfr_rtmt_taxa,
        ).limit(limit)

        if customer_id is not None:
            stmt = stmt.where(AccountsTable.customer_id == customer_id)

        rows = sess.execute(stmt).all()
        df = pd.DataFrame(rows, columns=[x for x in KMAP_ACCOUNTS.keys()])
        return df
    except Exception:
        sess.rollback()
        raise


# --------------------------------------------------------------------------------------
# Context Helpers
# --------------------------------------------------------------------------------------
def _ctx_to_dict(ctx: PensionContext | Dict[str, Any]) -> Dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        return ctx
    if is_dataclass(ctx):
        try:
            return asdict(ctx)
        except Exception:
            return getattr(ctx, "__dict__", {}) or {}
    return getattr(ctx, "__dict__", {}) or {}


# --- AgGrid 선택 행 파싱 헬퍼 -----------------------------------------------
def _extract_selected_rows(grid_res) -> list[dict]:
    rows = None
    if isinstance(grid_res, dict):
        rows = grid_res.get("selected_rows")
    elif hasattr(grid_res, "selected_rows"):
        rows = getattr(grid_res, "selected_rows")

    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records") if not rows.empty else []
    if isinstance(rows, list):
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append(r)
            elif isinstance(r, pd.Series):
                out.append(r.to_dict())
            else:
                try:
                    out.append(dict(r))
                except Exception:
                    pass
        return out
    return []


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _accounts_by_product_chart(df_accounts: pd.DataFrame):
    if df_accounts is None or df_accounts.empty:
        st.info("그래프를 표시할 데이터가 없습니다.")
        return

    req_cols = {"acnt_type", "acnt_evlu_amt"}
    if not req_cols.issubset(df_accounts.columns):
        st.info(f"필수 컬럼 부족: {req_cols} / 실제: {list(df_accounts.columns)}")
        return

    df_plot = df_accounts.copy()
    df_plot["acnt_evlu_amt"] = pd.to_numeric(df_plot["acnt_evlu_amt"], errors="coerce").fillna(0)

    if df_plot["acnt_evlu_amt"].sum() <= 0:
        st.info("그래프를 표시할 금액 데이터가 없습니다.")
        return

    chart_df = (
        df_plot.groupby("acnt_type", dropna=False, as_index=False)["acnt_evlu_amt"].sum()
        .sort_values("acnt_evlu_amt", ascending=True)
    )

    base = alt.Chart(chart_df).properties(height=280)
    bars = base.mark_bar(size=18).encode(
        y=alt.Y("acnt_type:N", title="상품유형", sort=None),
        x=alt.X("acnt_evlu_amt:Q", title="평가액 합계", axis=alt.Axis(format="~s")),
        color=alt.Color("acnt_type:N", legend=alt.Legend(title="상품유형")),
        tooltip=[
            alt.Tooltip("acnt_type:N", title="상품유형"),
            alt.Tooltip("acnt_evlu_amt:Q", title="평가액 합계", format=",.0f"),
        ],
    )
    text = base.mark_text(
        align="left", baseline="middle", dx=6
    ).encode(
        y="acnt_type:N",
        x="acnt_evlu_amt:Q",
        text=alt.Text("acnt_evlu_amt:Q", format=",.0f"),
    )

    st.altair_chart(bars + text, use_container_width=True)


# --------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------
def _render_customer_summary(row: pd.Series, labels: Dict[str, str]) -> Dict[str, Any]:
    fields = ["customer_name", "customer_id", "brth_dt", "cust_ivst_icln_grad_cd", "tot_asst_amt"]
    display: Dict[str, Any] = {}

    for f in fields:
        if f in row:
            label = labels.get(f, f)
            val = row[f]
            if f == "tot_asst_amt":
                try:
                    num = pd.to_numeric(val, errors="coerce")
                    val = f"{int(num):,}" if pd.notnull(num) else "-"
                except Exception:
                    pass
            if pd.isna(val):
                val = "-"
            display[label] = val

    rows_html = "".join(
        f"""
        <div class="kv-row">
            <div class="kv-key">{k}</div>
            <div class="kv-val">{v}</div>
        </div>
        """ for k, v in display.items()
    )

    st.markdown(
        f"""
        <style>
        .kv-card {{
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 10px 12px;
            background: linear-gradient(180deg, #fafbff 0%, #f6f7fb 100%);
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }}
        .kv-row {{
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 10px;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px dashed rgba(0,0,0,0.08);
        }}
        .kv-row:last-child {{ border-bottom: none; }}
        .kv-key {{
            color: #6b7280;
            font-size: 0.92rem;
        }}
        .kv-val {{
            color: #111827;
            font-weight: 600;
            font-size: 0.98rem;
            word-break: break-all;
        }}
        </style>
        <div class="kv-card">{rows_html}</div>
        """,
        unsafe_allow_html=True,
    )

    return display


def render_info_pane():
    st.subheader("고객/계좌 정보")

    try:
        sess = _get_session()
    except Exception as e:
        st.error("DB 세션 생성 실패(db_url 확인).")
        st.exception(e)
        return

    with st.spinner("고객 목록 로딩..."):
        try:
            df_customers = _load_customers(sess, q=None)
        except Exception as e:
            try: sess.rollback()
            except: pass
            st.error("고객 목록 조회 실패")
            st.exception(e)
            return

    if df_customers.empty:
        st.info("고객 데이터가 없습니다.")
        return

    if "customer_id" in df_customers.columns:
        df_customers["customer_id"] = df_customers["customer_id"].astype(str)

    df_customers["display"] = df_customers.apply(
        lambda r: f"{r.get('customer_name','-')} ({r.get('customer_id','-')})", axis=1
    )
    names = ["— 고객을 선택하세요 —"] + df_customers["display"].tolist()
    sel = st.selectbox("고객 선택 (검색 가능)", options=names, index=0, key="customer_select_box")

    # 상태 플래그
    if "accounts_grid_interacted" not in st.session_state:
        st.session_state["accounts_grid_interacted"] = False
    if "last_customer_for_grid" not in st.session_state:
        st.session_state["last_customer_for_grid"] = None

    if sel == "— 고객을 선택하세요 —":
        update_ctx(
            customer_id=None,
            customer=None,
            customer_display=None,
            accounts=list(),
            sim_params=dict()
        )
        st.info("고객을 선택하면 상세 정보와 계좌가 표시됩니다.")
        return

    sel_row = df_customers.loc[df_customers["display"] == sel]
    if sel_row.empty:
        st.warning("선택된 고객을 찾을 수 없습니다.")
        return
    sel_row = sel_row.iloc[0]
    selected_customer_id = str(sel_row["customer_id"])

    with st.spinner("계좌 로딩..."):
        try:
            df_accounts = _load_accounts(sess, customer_id=selected_customer_id)
        except Exception as e:
            try: sess.rollback()
            except: pass
            st.error("계좌 조회 실패")
            st.exception(e)
            df_accounts = pd.DataFrame(columns=[x for x in KMAP_ACCOUNTS.keys()])

    if "acnt_evlu_amt" in df_accounts.columns:
        df_accounts["acnt_evlu_amt"] = pd.to_numeric(df_accounts["acnt_evlu_amt"], errors="coerce").fillna(0)

    # 고객 요약 + 차트
    #st.markdown("---")
    colL, colR = st.columns([0.45, 0.55])
    with colL:
        customer_display_kor = _render_customer_summary(sel_row, labels=KMAP_CUSTOMERS)

    #st.markdown("---")
    st.markdown("**계좌 목록**")
    grid_res = aggrid_table(
        df_accounts,
        key="accounts_grid",
        selection_mode="multiple",
        enable_header_checkbox=True,
        height=300,
        fit_columns_on_load=False,
        allow_horizontal_scroll=True,
        display_labels=KMAP_ACCOUNTS,
        select_all_on_load=True,            # 최초 렌더에서 전체 선택
        select_all_on_data_change=True,     # 고객 바꿔서 df가 교체될 때마다 전체 선택
        select_filtered_only=False,         # 필터 무관 전체 선택
    )
    selected_rows = _extract_selected_rows(grid_res)
    selected_df = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame()

    # 고객이 바뀌면 reset
    if st.session_state["last_customer_for_grid"] != selected_customer_id:
        st.session_state["accounts_grid_interacted"] = False
        st.session_state["last_customer_for_grid"] = selected_customer_id

    # 상호작용 없을 땐 전체 선택, 이후에는 실제 선택 반영
    if not st.session_state["accounts_grid_interacted"]:
        selected_df = df_accounts.copy()
    else:
        # 사용자가 일부/모두 해제한 경우 그대로 반영 → 비면 context.accounts=[]
        pass

    # 플래그 on (한번이라도 선택 변화가 있으면)
    if grid_res is not None:
        st.session_state["accounts_grid_interacted"] = True

    with colR:
        st.markdown("**상품유형별 평가액 합계 (선택 반영)**")
        _accounts_by_product_chart(selected_df)

    customer_py = {k: (v.item() if hasattr(v, "item") else v) for k, v in sel_row.to_dict().items()}
    accounts_dict = selected_df.to_dict(orient="records")
    update_ctx(
        customer_id=selected_customer_id,
        customer=customer_py,
        customer_display=customer_display_kor,
        accounts=accounts_dict,       
    )

    # with st.expander("컨텍스트 미리보기 (좌측, 선택 반영)", expanded=False):
    #     from ui.utils import get_ctx_dict
    #     st.json(get_ctx_dict())
