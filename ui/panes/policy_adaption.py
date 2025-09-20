# ui/pane/policy_timeline.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from pathlib import Path

from ui.components.policy_engine import load_policies, evaluate_policies

MARKDOWN_DIR = Path("resources/markdown")  # 필요 시 변경

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _format_current(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if isinstance(val, (int, float)):
        try:
            return f"{val:,.0f}" if float(val).is_integer() else f"{val:,.2f}"
        except Exception:
            return str(val)
    return str(val)

def _coerce_inputs(
    customer_dict: Optional[Dict[str, Any]],
    accounts_list: Optional[List[Dict[str, Any]]],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Dict → (pd.Series, pd.DataFrame) 변환
    - customer_dict: {"customer_id": "...", "brth_dt": "...", ...}
    - accounts_list: [{"account_id": "...", "expd_dt": "...", ...}, ...]
    """
    cust = pd.Series(customer_dict or {})
    acc = pd.DataFrame(accounts_list or [])
    if acc.empty:
        acc = pd.DataFrame(columns=[
            "account_id","customer_id","acnt_type","prd_type_cd","acnt_bgn_dt",
            "expd_dt","etco_dt","rtmt_dt","midl_excc_dt","acnt_evlu_amt",
            "copt_year_pymt_amt","other_txtn_ecls_amt","rtmt_incm_amt",
            "icdd_amt","user_almt_amt","sbsr_almt_amt","utlz_erng_amt","dfr_rtmt_taxa"
        ])
    return cust, acc

def _palette() -> List[str]:
    # 고정 팔레트: 계좌별 구분 컬러
    return [
        "#418FDE",  # brand blue (PANTONE 279C)
        "#603314",  # brand brown (PANTONE 732C)
        "#7F56D9",
        "#12B76A",
        "#EAAA08",
        "#F04438",
        "#0EA5E9",
        "#8B5CF6",
        "#22C55E",
        "#F59E0B",
    ]

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def render_policy_adption_section(
    customer: Dict[str, Any],
    accounts: List[Dict[str, Any]],
) -> None:
    """
    Dict 인자를 받습니다.
    - customer: 단일 고객 dict
    - accounts: 해당 고객의 계좌 dict 리스트
    """
    customer_row, accounts_df = _coerce_inputs(customer, accounts)

    # ── 정책 매칭(기존 표 형식 유지; 필요 시 이후에 별도 탭으로 분리 가능)
    #st.markdown("### 📑 정책 매칭 (조건 ↔ 현재값 ↔ 판정)")
    try:
        policies = load_policies(MARKDOWN_DIR)
        if not policies:
            st.info("정책 .md에서 policy 블록을 찾지 못했습니다. (```yaml policy: ... ``` 형태)")
            return

        evaled = evaluate_policies(policies, customer_row, accounts_df)
        top = evaled[:3] if len(evaled) > 3 else evaled

        for item in top:
            with st.container(border=True):
                title = item.get("title") or item.get("policy_id")
                anchor = item.get("anchor")
                file_path = item.get("file")
                header = f"**{title}**"
                if anchor:
                    header += f"  \n*근거 위치: `{anchor}`*"
                st.markdown(header)

                rows = []
                for c in item.get("conditions", []):
                    verdict = "✅ 충족" if c.get("result") else "—"
                    rows.append({
                        "조건ID": c.get("id"),
                        "필드": c.get("field"),
                        "연산": c.get("op"),
                        "기준값": c.get("value"),
                        "현재값": _format_current(c.get("current")),
                        "판정": verdict
                    })
                df = pd.DataFrame(rows, columns=["조건ID", "필드", "연산", "기준값", "현재값", "판정"])
                st.dataframe(df, use_container_width=True, hide_index=True)

                eff = item.get("effects", {}) or {}
                eff_txts = []
                for k in ("eligible", "caution", "info", "ineligible"):
                    ids = eff.get(k, [])
                    if ids:
                        eff_txts.append(f"{k}: {', '.join(ids)}")
                if eff_txts:
                    st.caption("효과: " + " | ".join(eff_txts))

                with st.expander("🔍 근거 요약 보기"):
                    any_snippet = False
                    for c in item.get("conditions", []):
                        snip = c.get("snippet")
                        if snip:
                            any_snippet = True
                            st.markdown(f"**• {c.get('id')}**  \n{snip}")
                    if not any_snippet:
                        st.caption("이 정책에는 스니펫이 정의되어 있지 않습니다.")
                st.caption(f"원문: `{file_path}`")

        if len(evaled) > len(top):
            st.link_button("더 많은 정책 결과 보기", "javascript:window.scrollTo(0,0);")
    except Exception as ex:
        st.warning(f"정책 판정 중 오류: {ex}")



