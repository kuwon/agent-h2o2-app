# ui/panes/sim.py
import streamlit as st
import pandas as pd
from typing import Any, Dict
from dataclasses import asdict, is_dataclass

def _ctx_to_dict(ctx: Any) -> Dict[str, Any]:
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

def render_sim_pane(ctx_obj: Any):
    """현재 컨텍스트를 기반으로 간단한 시뮬레이션 데모."""
    st.markdown("#### 시뮬레이션")
    ctx = _ctx_to_dict(ctx_obj)

    if not ctx or not ctx.get("accounts"):
        st.info("컨텍스트에 계좌가 없습니다. 좌측에서 고객을 선택하세요.")
        return

    df = pd.DataFrame(ctx["accounts"])
    if "acnt_evlu_amt" in df.columns:
        df["acnt_evlu_amt"] = pd.to_numeric(df["acnt_evlu_amt"], errors="coerce").fillna(0)

    # 파라미터 예시
    years = st.slider("기간(년)", 1, 30, 10, 1)
    rate  = st.slider("연 수익률(%)", 0.0, 15.0, 4.0, 0.5)

    current = float(df.get("acnt_evlu_amt", pd.Series([0])).sum())
    proj = current * ((1 + rate/100.0) ** years)
    st.metric("현재 평가액 합계", f"{int(current):,}")
    st.metric(f"{years}년 후 예상", f"{int(proj):,}")

    st.caption("※ 데모: 실제 규정/세제/입출금 반영 로직은 도메인에 맞게 확장하세요.")
