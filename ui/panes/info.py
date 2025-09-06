
import streamlit as st, pandas as pd
from ui.components.aggrid import aggrid_table
from ui.state import PensionContext
def _ensure_context():
    ctx: PensionContext = st.session_state["context"]
    if not ctx.accounts: ctx.accounts=[{"account_id":"A1","prd_type_cd":"DC","evlu":42800000},{"account_id":"A2","prd_type_cd":"IRP","evlu":12500000}]
    if not ctx.dc_contracts: ctx.dc_contracts=[{"ctrt_no":"D-001","status":"ACTIVE","evlu_acca_smtl_amt":42800000,"sst_join_dt":"2019-03-01"}]
def render_info_pane():
    st.subheader("고객/계좌 정보"); _ensure_context(); ctx: PensionContext = st.session_state["context"]
    st.text_input("고객 ID", key="customer_id_input", value=ctx.customer_id or "C123")
    if st.button("고객 선택/변경"): ctx.customer_id = st.session_state.get("customer_id_input")
    st.markdown("**계좌 요약**"); aggrid_table(pd.DataFrame(ctx.accounts))
    st.markdown("**DC 계약 요약**"); aggrid_table(pd.DataFrame(ctx.dc_contracts))
    st.divider()
    if st.button("▶ 시뮬레이션 화면으로", use_container_width=True): st.session_state.left_view="sim"; st.rerun()
