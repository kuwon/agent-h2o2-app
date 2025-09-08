# ui/panes/context_builder.py
from __future__ import annotations
import json
import streamlit as st
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from workspace.utils.db_key_eng_kor import KMAP_ACCOUNTS, KMAP_CUSTOMERS
from ui.state import PensionContext
from ui.utils import _ctx_to_dict_any


def _ctx_to_dict(ctx: PensionContext | Dict[str, Any]) -> Dict[str, Any]:
    if ctx is None: return {}
    if isinstance(ctx, dict): return ctx
    if is_dataclass(ctx):
        try: return asdict(ctx)
        except Exception: return getattr(ctx, "__dict__", {}) or {}
    return getattr(ctx, "__dict__", {}) or {}

def _translate_customer_to_kor(customer: Dict[str, Any]) -> Dict[str, Any]:
    # 영문키 → 한글 라벨
    out = {}
    for k, v in (customer or {}).items():
        out[KMAP_CUSTOMERS.get(k, k)] = v
    return out

def _translate_accounts_to_kor(accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kor = []
    for row in accounts or []:
        kor.append({KMAP_ACCOUNTS.get(k, k): v for k, v in row.items()})
    return kor

def render_context_inline(expanded: bool = False) -> None:
    if "context" not in st.session_state or st.session_state["context"] is None:
        st.session_state["context"] = PensionContext()

    ctx_dict = _ctx_to_dict_any(st.session_state["context"])

    with st.expander("🧩 Context Builder (편집 가능)", expanded=expanded):
        # 보기 언어: 한글/영문
        view_lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True, key="ctx_view_lang")

        if view_lang == "한국어":
            # ✅ 한글로 변환된 view
            view_obj = {
                "고객번호": ctx_dict.get("customer_id"),
                "고객": _translate_customer_to_kor(ctx_dict.get("customer")),
                "계좌": _translate_accounts_to_kor(ctx_dict.get("accounts") or []),
                "시뮬레이션설정": ctx_dict.get("sim_params", {}),
            }
            st.json(view_obj)
            st.caption("보기 전용입니다. 편집은 아래 '편집 모드'에서 영문 JSON으로 진행하세요.")
        else:
            st.json(ctx_dict)

        edit_mode = st.toggle("편집 모드", value=False, key="context_edit_mode_inline")

        if edit_mode:
            st.caption("편집은 영문 키 기준 JSON으로만 지원합니다. (팀/에이전트 호환 위해)")
            text = st.text_area(
                "Context JSON (English keys)",
                value=json.dumps(ctx_dict, ensure_ascii=False, indent=2),
                height=280,
                key="context_edit_text_inline",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("저장", type="primary", use_container_width=True, key="ctx_inline_save"):
                    try:
                        new_obj = json.loads(text)
                        # customer_id는 문자열로 유지
                        if "customer_id" in new_obj and new_obj["customer_id"] is not None:
                            new_obj["customer_id"] = str(new_obj["customer_id"])
                        st.session_state["context"] = new_obj
                        st.success("컨텍스트가 업데이트되었습니다.")
                    except Exception as e:
                        st.error(f"JSON 파싱 오류: {e}")
            with c2:
                if st.button("되돌리기", use_container_width=True, key="ctx_inline_revert"):
                    st.info("편집을 되돌렸습니다.")
