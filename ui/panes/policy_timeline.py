# ui/pane/policy_timeline.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import hashlib
import pandas as pd
import streamlit as st
from pathlib import Path

from ui.components.policy_engine import load_policies, evaluate_policies, build_timeline

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

def _color_for_account(aid: str, acnt_type: Optional[str]) -> str:
    # account_id + acnt_type 조합으로 안정적 색상 선택
    base = f"{aid}|{acnt_type or ''}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_palette())
    return _palette()[idx]

def _build_fishbone_html(
    events: List[Dict[str, Any]],
    account_type_map: Dict[str, str],
) -> str:
    """
    Fishbone: 중앙 라인 위/아래로 번갈아 이벤트 배치.
    각 칸은 그리드의 column에 대응. 계좌 이벤트는 색상 라벨.
    표시 텍스트: [일시] [계좌번호 | 구분] [이벤트종류]
    """
    if not events:
        return "<p>표시할 타임라인 이벤트가 없습니다.</p>"

    # 날짜 순으로 들어온 events를 그대로 좌→우로 배치
    n = len(events)

    # 각 이벤트의 렌더 블럭 생성
    def render_event(i: int, ev: Dict[str, Any]) -> str:
        dt = ev.get("date")
        date_txt = "-" if not dt else dt.strftime("%Y-%m-%d")
        label = ev.get("label") or "-"
        kind = ev.get("kind") or "-"
        meta = ev.get("meta") or {}
        aid = meta.get("account_id")
        acnt_type = account_type_map.get(aid, None) if aid else None

        # 색상: 계좌 이벤트면 계좌별 색, 생년월일 등 고객 이벤트면 중립
        if aid:
            color = _color_for_account(aid, acnt_type)
            acct_pill = f"""<span class="acct-pill" style="background:{color}1A;color:{color};border:1px solid {color}33">
                {aid} <span class="sep">|</span> {acnt_type or '-'}
            </span>"""
        else:
            color = "#6B7280"  # neutral gray
            acct_pill = f"""<span class="acct-pill" style="background:#eee;color:#374151;border:1px solid #e5e7eb">
                -
            </span>"""

        # 점/스파인 컬러도 계좌 컬러(고객 이벤트는 중립)
        dot_style = f"background:{color};border-color:{color}55"

        # 본문(위/아래 카드)
        body = f"""
        <div class="ev-card">
            <div class="ev-date">{date_txt}</div>
            <div class="ev-acct">{acct_pill}</div>
            <div class="ev-kind">{label}</div>
        </div>
        """

        # 하나의 column cell
        return f"""
        <div class="fb-col" style="grid-column:{i+1}">
            <div class="fb-branch">
                <span class="fb-dot" style="{dot_style}"></span>
                <span class="fb-stick" style="border-color:{color}40"></span>
            </div>
            {body}
        </div>
        """

    top_cells = []
    bot_cells = []
    for i, ev in enumerate(events):
        cell_html = render_event(i, ev)
        # 짝수 인덱스는 위쪽, 홀수는 아래쪽
        if i % 2 == 0:
            top_cells.append(cell_html)
        else:
            bot_cells.append(cell_html)

    # grid-template-columns: repeat(n, minmax(140px, 1fr))
    grid_cols = f"repeat({n}, minmax(140px, 1fr))"

    return f"""
<style>
.fishbone {{
  width: 100%;
  margin: 8px 0 16px;
}}
.fishbone .fb-top, .fishbone .fb-bottom {{
  display: grid;
  grid-template-columns: {grid_cols};
  gap: 0.75rem;
  align-items: end;
}}
.fishbone .fb-center {{
  position: relative;
  height: 2px;
  background: linear-gradient(90deg, rgba(0,0,0,0.08), rgba(0,0,0,0.12), rgba(0,0,0,0.08));
  margin: 10px 0;
}}
.fb-col {{
  display: flex;
  flex-direction: column;
  align-items: center;
}}
.fb-branch {{
  position: relative;
  height: 28px;
  width: 2px;
  border-left: 2px dashed rgba(0,0,0,0.12);
  margin-bottom: 6px;
}}
.fb-dot {{
  position: absolute;
  top: -7px;
  left: -6px;
  width: 12px; height: 12px;
  border-radius: 999px;
  border: 2px solid rgba(0,0,0,0.15);
  background: #999;
}}
.fb-stick {{
  position: absolute;
  bottom: -6px;
  left: -1px;
  display: inline-block;
  height: 12px;
  border-left: 2px solid rgba(0,0,0,0.15);
}}
.ev-card {{
  background: #fff;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  border-radius: 10px;
  padding: 6px 8px;
  text-align: center;
  min-width: 120px;
}}
.ev-date {{
  font-size: 12px;
  color: #6B7280;
  margin-bottom: 2px;
}}
.acct-pill {{
  display: inline-block;
  font-size: 11px;
  line-height: 1;
  padding: 5px 8px;
  border-radius: 999px;
  margin: 0 0 4px 0;
  white-space: nowrap;
}}
.ev-acct {{ margin-bottom: 2px; }}
.ev-kind {{ font-weight: 600; font-size: 13px; color: #111827; }}
.sep {{ opacity: 0.6; padding: 0 4px; }}
@media (prefers-color-scheme: dark) {{
  .ev-card {{
    background: #0b0f17;
    border-color: rgba(255,255,255,0.08);
    box-shadow: 0 2px 14px rgba(0,0,0,0.35);
  }}
  .ev-kind {{ color: #e5e7eb; }}
  .ev-date {{ color: #9CA3AF; }}
}}
</style>

<div class="fishbone">
  <div class="fb-top">
    {''.join(top_cells)}
  </div>
  <div class="fb-center"></div>
  <div class="fb-bottom">
    {''.join(bot_cells)}
  </div>
</div>
"""

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def render_policy_and_timeline_section(
    customer: Dict[str, Any],
    accounts: List[Dict[str, Any]],
) -> None:
    """
    Dict 인자를 받습니다.
    - customer: 단일 고객 dict
    - accounts: 해당 고객의 계좌 dict 리스트
    """
    customer_row, accounts_df = _coerce_inputs(customer, accounts)

    # ── 개인 타임라인 (Fishbone)
    st.markdown("### 🧭 개인 타임라인")
    try:
        events = build_timeline(customer_row, accounts_df)

        # account_id -> acnt_type 매핑 (fishbone에서 보여줄 구분값)
        acc_type_map = {}
        if "account_id" in accounts_df.columns and "acnt_type" in accounts_df.columns:
            acc_type_map = (
                accounts_df[["account_id", "acnt_type"]]
                .dropna(subset=["account_id"])
                .drop_duplicates()
                .set_index("account_id")["acnt_type"]
                .to_dict()
            )

        html = _build_fishbone_html(events, acc_type_map)
        st.markdown(html, unsafe_allow_html=True)
        st.divider()
    except Exception as ex:
        st.warning(f"타임라인 생성 중 오류: {ex}")

    # ── 정책 매칭 (기존 표 그대로 유지; 필요 시 이후에 fishbone만 별도로 유지해도 됨)
    st.markdown("### 📑 정책 매칭 (조건 ↔ 현재값 ↔ 판정)")
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

