# ui/pane/policy_timeline_vertical.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import pandas as pd
import streamlit as st
from pathlib import Path
from ui.components.policy_engine import build_timeline

MARKDOWN_DIR = Path("resources/markdown")  # 필요 시 변경

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
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
    return [
        "#418FDE", "#603314", "#7F56D9", "#12B76A", "#EAAA08",
        "#F04438", "#0EA5E9", "#8B5CF6", "#22C55E", "#F59E0B",
    ]

def _color_for_account(aid: str, acnt_type: Optional[str]) -> str:
    base = f"{aid}|{acnt_type or ''}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_palette())
    return _palette()[idx]

def _icon_for_kind(kind: Optional[str]) -> str:
    k = (kind or "").lower()
    if k in ("birth","brth","birthday","생년월일"): return "🎂"
    if k in ("begin","open","개설","계좌개설"): return "🏦"
    if k in ("mid","중간정산","중정","mid_settlement"): return "🛠️"
    if k in ("retire","퇴직","retirement"): return "👋"
    if k in ("maturity","만기"): return "⏰"
    return "📌"

# ─────────────────────────────────────────────
# HTML Builder (Vertical Zig-Zag)
# ─────────────────────────────────────────────
def _build_vertical_timeline_html(
    events: List[Dict[str, Any]],
    account_type_map: Dict[str, str]
) -> str:
    if not events:
        return "<p>표시할 이벤트가 없습니다.</p>"

    rows_html: List[str] = []
    for i, ev in enumerate(events):
        dt = ev.get("date")
        date_txt = dt.strftime("%Y-%m-%d") if dt else "-"
        label = ev.get("label") or "-"
        kind = ev.get("kind") or "-"
        icon = _icon_for_kind(kind)
        meta = ev.get("meta") or {}
        aid = meta.get("account_id")
        acnt_type = account_type_map.get(aid) if aid else None
        color = _color_for_account(aid, acnt_type) if aid else "#6B7280"
        side = "left" if i % 2 == 0 else "right"  # 지그재그

        # 계좌 표시 pill
        acct_text = f'{aid or "-"}'
        if acnt_type:
            acct_text += ' <span class="sep">|</span> ' + str(acnt_type)
        acct_pill = (
            '<span class="acct-pill" '
            f'style="background:{color}1A;color:{color};border:1px solid {color}33">'
            f'{acct_text}'
            '</span>'
        )

        # 카드(보더/팔 색은 row의 --accent로 제어)
        content = (
            '<div class="vt-content">'
            f'  <div class="vt-date">{date_txt}</div>'
            f'  <div class="vt-acct">{acct_pill}</div>'
            f'  <div class="vt-kind"><span class="vt-icon">{icon}</span>'
            f'    <span class="vt-text">{label}</span></div>'
            '</div>'
        )
        card = f'<div class="vt-card" data-side="{side}">{content}</div>'

        # 중앙칸에 "연결선"만 그림(마커 제거) → 좌/우 모두 스파인까지 정확히 붙음
        left_cell  = f'<div class="cell left">{card if side=="left" else ""}</div>'
        mid_cell   = '<div class="cell mid"></div>'  # 마커 제거
        right_cell = f'<div class="cell right">{card if side=="right" else ""}</div>'

        row = (
            f'<div class="vt-row {side}" style="--accent:{color};--i:{i}">'
            f'{left_cell}{mid_cell}{right_cell}'
            '</div>'
        )
        rows_html.append(row)

    css = """
<style>
.vt-wrap { width: 100%; }

/* 3열 Grid 컨테이너 */
.vt-grid {
  position: relative;
  display: grid;
  grid-auto-rows: minmax(68px, auto);
  row-gap: 18px;
}

/* 중앙 스파인(컨테이너 전체 높이) */
.vt-grid::before {
  content: "";
  position: absolute;
  left: 50%;
  transform: translateX(-2px);
  top: 0; bottom: 0;
  width: 4px;
  background: #418FDE;
  border-radius: 2px;
  box-shadow: 0 0 0 1px rgba(65,143,222,0.35), 0 0 18px rgba(65,143,222,0.28);
}

/* 각 행(좌/중앙/우) */
.vt-row {
  display: grid;
  grid-template-columns: 1fr 46px 1fr; /* 중앙 46px 고정 */
  align-items: center;
  gap: 8px;
  position: relative;
}

/* 중앙 칸: 좌/우에 따라 스파인까지 연결선 그리기 */
.vt-row .cell.mid { position: relative; }
.vt-row.left  .cell.mid::before,
.vt-row.right .cell.mid::before {
  content: "";
  position: absolute;
  top: 50%;
  height: 2px;
  width: 50%;              /* 중앙칸의 절반 → 스파인 중심까지 도달 */
  background: var(--accent, #A3A3A3);
  transform: translateY(-50%);
}
.vt-row.left  .cell.mid::before  { left: 0; }  /* 카드 → 오른쪽(스파인)으로 선 연결 */
.vt-row.right .cell.mid::before  { right: 0; } /* 카드 → 왼쪽(스파인)으로 선 연결 */

/* 카드 칸 */
.vt-row .cell.left,
.vt-row .cell.right {
  display: flex;
  align-items: center;
}
.vt-row .cell.left  { justify-content: flex-end; }  /* 왼쪽은 오른쪽 정렬 */
.vt-row .cell.right { justify-content: flex-start; }/* 오른쪽은 왼쪽 정렬 */

/* 카드 */
.vt-card { position: relative; max-width: 96%; }
.vt-content {
  background: #fff;
  border: 2px solid var(--accent, #CBD5E1);
  border-radius: 12px;
  padding: 8px 12px;
  min-width: 160px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* (기존) 카드 자체의 팔 제거 → 중앙칸에서 그리도록 변경 */
/* .vt-content::after { ... } 제거 */

/* 타이포/배지 */
.vt-date { font-size: 12px; color: #6B7280; margin-bottom: 3px; }
.vt-acct { margin-bottom: 4px; }
.acct-pill {
  display: inline-block;
  font-size: 11px;
  line-height: 1;
  padding: 4px 6px;
  border-radius: 999px;
  white-space: nowrap;
}
.vt-kind {
  display: inline-flex; align-items: center; gap: 6px; justify-content: center;
  font-weight: 700; font-size: 13px; color: #111827; letter-spacing: 0.1px;
}
.vt-icon { font-size: 16px; line-height: 1; transform: translateY(1px); }
.sep { opacity: .6; padding: 0 4px; }

/* 등장 애니메이션 */
@keyframes floatInLeft  { from { opacity:0; transform: translateX(-12px) scale(.98); } to { opacity:1; transform:none; } }
@keyframes floatInRight { from { opacity:0; transform: translateX( 12px) scale(.98); } to { opacity:1; transform:none; } }
.vt-row.left  .vt-card .vt-content  { animation: floatInLeft  .42s ease-out both;  animation-delay: calc(var(--i, 0) * 60ms); }
.vt-row.right .vt-card .vt-content  { animation: floatInRight .42s ease-out both;  animation-delay: calc(var(--i, 0) * 60ms); }

/* 다크 모드 보정 */
@media (prefers-color-scheme: dark) {
  .vt-content {
    background:#0b0f17;
    border-color: color-mix(in oklab, var(--accent, #6B7280) 75%, #0b0f17);
    box-shadow: 0 2px 16px rgba(0,0,0,0.45);
  }
  .vt-kind { color: #E6E9EE; }
  .vt-date { color:#AEB6C2; }
  .vt-grid::before {
    background: #5AA2E4;
    box-shadow: 0 0 0 1px rgba(90,162,228,0.55), 0 0 22px rgba(90,162,228,0.40);
  }
}
</style>
"""

    html = css + '<div class="vt-wrap"><div class="vt-grid">' + "".join(rows_html) + "</div></div>"
    return html

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def render_timeline_section(
    customer: Dict[str, Any],
    accounts: List[Dict[str, Any]],
) -> None:
    """
    Dict 인자를 받습니다.
    """
    customer_row, accounts_df = _coerce_inputs(customer, accounts)

    # ── 개인 타임라인 (Vertical Zig-Zag)
    #st.markdown("### 🧭 개인 타임라인 (세로)")
    try:
        events = build_timeline(customer_row, accounts_df)

        # account_id -> acnt_type 매핑
        acc_type_map: Dict[str, str] = {}
        if not accounts_df.empty and "account_id" in accounts_df.columns and "acnt_type" in accounts_df.columns:
            acc_type_map = (
                accounts_df[["account_id", "acnt_type"]]
                .dropna(subset=["account_id"])
                .drop_duplicates()
                .set_index("account_id")["acnt_type"]
                .to_dict()
            )

        html = _build_vertical_timeline_html(events, acc_type_map)
        st.markdown(html, unsafe_allow_html=True)
        #st.divider()
    except Exception as ex:
        st.warning(f"타임라인 생성 중 오류: {ex}")
