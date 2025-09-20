import streamlit as st
from .info import render_info_pane
from .sim import render_sim_pane
from .policy_adaption import render_policy_adaption_section
from .vertical_timeline import render_timeline_section
from ui.utils import get_ctx_dict

def render_left_pane(view: str = "info"):
    if view == "sim":
        render_sim_pane()
    else:
        render_info_pane()
        try:
            ctx = get_ctx_dict()  # context에 저장된 현재 선택 고객/계좌 등
            customer = ctx.get("customer", {}) or {}
            accounts = ctx.get("accounts", []) or []

            # Dict 그대로 전달
            #with st.container():
            with st.expander("## 🧭 개인 타임라인", expanded=True):
                render_timeline_section(customer=customer, accounts=accounts)
            with st.expander("## 📑 정책 매칭 (조건 ↔ 현재값 ↔ 판정)", expanded=True):
                render_policy_adaption_section(customer=customer, accounts=accounts)
                
        except Exception as ex:
            st.warning(f"정책/타임라인 섹션 렌더링 중 오류: {ex}")        
