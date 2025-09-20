# ui/pane/policy_apply.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.policy_engine import load_policies, evaluate_policies

MARKDOWN_DIR = Path("resources/markdown")  # 필요 시 경로 조정

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# NEW: dict/객체 겸용 안전 접근 유틸
# ─────────────────────────────────────────────
def _sg(obj: Any, key: str, default: Any = None) -> Any:
    """safe-get: dict면 get, 객체면 getattr, 둘 다 아니면 default"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _as_dict(obj: Any) -> Dict[str, Any]:
    """객체를 dict처럼 평탄화(가능하면 __dict__)"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # dataclass/일반 객체 대응
    d = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        # 메서드/콜러블 제외
        if callable(val):
            continue
        d[attr] = val
    return d


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

def _has_selected_customer(customer: Dict[str, Any]) -> bool:
    # 고객 미선택 기준: dict 비었거나, customer_id가 없거나 빈 값
    if not customer:
        return False
    cid = str(customer.get("customer_id") or "").strip()
    return cid != ""

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def render_policy_adaption_section(
    customer: Dict[str, Any],
    accounts: List[Dict[str, Any]],
    *,
    max_policies: int = 3,
) -> None:
    """
    정책 적용 섹션 렌더러 (Dict 입력)
    - 고객 미선택 시: 아무 계산/표시 없이 안내 문구만 노출
    - 근거 요약(스니펫)을 상단에, 계산/판정 상세는 Expander 안에
    - Policy/Condition이 '객체'여도 안전하게 렌더
    """
    st.markdown("### 📑 정책 적용")

    # 1) 고객 미선택이면 바로 반환 (값 노출 금지)
    if not _has_selected_customer(customer):
        st.info("좌측에서 고객을 선택하면 정책 적용 결과를 보여드릴게요.")
        return

    # 2) 입력 변환
    customer_row, accounts_df = _coerce_inputs(customer, accounts)

    # 3) 정책 로딩
    try:
        policies = load_policies(MARKDOWN_DIR)
    except Exception as ex:
        st.warning(f"정책(.md) 로딩 오류: {ex}")
        return

    if not policies:
        st.info("정책 .md에서 policy 블록을 찾지 못했습니다. (```yaml policy: ... ``` 형태)")
        return

    # 4) 정책 평가
    try:
        evaled = evaluate_policies(policies, customer_row, accounts_df)
    except Exception as ex:
        st.warning(f"정책 판정 중 오류: {ex}")
        return

    # 상위만 노출
    items = evaled[:max_policies] if len(evaled) > max_policies else evaled

    for item in items:
        # dict/객체 모두 허용
        title  = _sg(item, "title") or _sg(item, "policy_id") or _sg(item, "id") or "정책"
        anchor = _sg(item, "anchor")
        file_path = _sg(item, "file") or _sg(item, "_file")
        conditions = _sg(item, "conditions") or []

        with st.container(border=True):
            header = f"**{title}**"
            if anchor:
                header += f"  \n*근거 위치: `{anchor}`*"
            st.markdown(header)

            # ───────── where(필터) 뱃지 (있을 때만) ─────────
            where_badges = []
            for c in conditions:
                w = _sg(c, "where")
                if isinstance(w, dict) and w:
                    pairs = []
                    for k, v in w.items():
                        if isinstance(v, (list, tuple)):
                            pairs.append(f"{k}: {', '.join(map(str, v))}")
                        else:
                            pairs.append(f"{k}: {v}")
                    where_badges.append(" / ".join(pairs))
            if where_badges:
                st.caption("적용 범위: " + " | ".join(sorted(set(where_badges))))

            # ───────── 근거 요약(우선 노출) ─────────
            pos_snips, other_snips = [], []
            for c in conditions:
                snip   = _sg(c, "snippet") or _sg(item, "snippets", {}).get(_sg(c, "id"), None) if isinstance(_sg(item, "snippets"), dict) else _sg(c, "snippet")
                result = bool(_sg(c, "result", False))
                cid    = _sg(c, "id") or _sg(c, "name") or ""
                if snip:
                    (pos_snips if result else other_snips).append((cid, snip))

            if pos_snips or other_snips:
                st.markdown("#### 🔍 근거 요약")
                if pos_snips:
                    for cid, snip in pos_snips:
                        st.markdown(f"- **{cid}**: {snip}")
                if (not pos_snips) and other_snips:
                    for cid, snip in other_snips:
                        st.markdown(f"- {cid}: {snip}")
            else:
                st.caption("이 정책에는 요약 스니펫이 정의되어 있지 않습니다.")

            # 효과 요약
            eff = _sg(item, "effects") or {}
            if isinstance(eff, dict):
                eff_txts = []
                for k in ("eligible", "caution", "info", "ineligible"):
                    ids = eff.get(k, [])
                    if ids:
                        eff_txts.append(f"{k}: {', '.join(map(str, ids))}")
                if eff_txts:
                    st.caption("효과: " + " | ".join(eff_txts))

            if file_path:
                st.caption(f"원문: `{file_path}`")

            # ───────── 계산/판정 상세(Expander 안) ─────────
            with st.expander("계산/판정 상세 보기"):
                rows = []
                for c in conditions:
                    cid    = _sg(c, "id") or _sg(c, "name") or ""
                    field  = _sg(c, "field")
                    op     = _sg(c, "op")
                    value  = _sg(c, "value")
                    cur    = _sg(c, "current")
                    result = bool(_sg(c, "result", False))
                    verdict = "✅ 충족" if result else "—"

                    rows.append({
                        "조건ID": cid,
                        "필드": field,
                        "연산": op,
                        "기준값": value,
                        "현재값": _format_current(cur),
                        "판정": verdict,
                    })
                if rows:
                    df = pd.DataFrame(rows, columns=["조건ID", "필드", "연산", "기준값", "현재값", "판정"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("표시할 조건이 없습니다.")

    if len(evaled) > len(items):
        st.link_button("더 많은 정책 결과 보기", "javascript:window.scrollTo(0,0);")