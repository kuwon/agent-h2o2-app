# st_sample.py
# -*- coding: utf-8 -*-
# 퇴직연금 RAG + 시뮬레이터
# - 고객명 검색 + 단건 요약 카드 + 파이차트 (범례 숨김, 내부 라벨)
# - 계좌 전체 + DC 계약 그리드 추가
# - 왼쪽 컨텍스트 변경 → 오른쪽 JSON 자동 갱신(st.rerun)
# - run_pension_simulator: dummy, DEFAULT_PARAM_SCHEMA: notes만

import os
import re
import json
import time
import traceback
from typing import Any, Dict, Optional, List
from urllib.parse import quote_plus, urlencode
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import components
import plotly.express as px

# AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid, GridUpdateMode

# SQLAlchemy / pgvector
from sqlalchemy.sql import bindparam
from sqlalchemy import create_engine, text, event
import pgvector.sqlalchemy
try:
    from pgvector.psycopg import register_vector
except Exception:
    register_vector = None

# agno (더미 구성)
from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.agent import AgentKnowledge


# ==================== Page Config & Styles ====================
st.set_page_config(page_title="한투 퇴직마스터", layout="wide")
st.markdown("""
<style>
.main .block-container { padding-top: .2rem; }
.center-title { text-align:center; margin: .3rem 0 .7rem 0; font-size: 1.8rem; font-weight: 700; }
.panel-soft { padding: 12px 14px; border-radius: 12px; background: #ffffff;
  border: 1px solid rgba(0,0,0,.06); box-shadow: 0 1px 2px rgba(0,0,0,.03); }
.panel-soft.flush-top { padding-top: 0; }
.panel-soft > :first-child { margin-top: 0 !important; }
.v-sep { border-left: 1px solid #e9ecef; height: calc(100vh - 180px); margin: 8px 6px; }
.kv { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 10px 0; }
.kv .item { background:#f8f9fa; border:1px solid #e9ecef; border-radius:10px; padding:8px 10px; font-size:.9rem; }
.kv .k { color:#666; margin-right:6px; }
</style>
<div class="center-title">한투 퇴직마스터</div>
""", unsafe_allow_html=True)


# ==================== Constants ====================
DEFAULT_PARAM_SCHEMA: Dict[str, Any] = {"notes": ""}

GRID_KEYS = {
    "cust": "grid_customer_v3",
    "acct": "grid_acct_v3",
    "dc": "grid_dc_v3",
}

# 한/영 라벨 맵 (표시: 한글, 내부: 영문)
KMAP_CUSTOMER = {
    "customer_id": "고객 번호",
    "customer_name": "고객 이름",
    "brth_dt": "생년월일",
    "age_band": "연령대",
}
KMAP_ACCOUNT = {
    "account_id": "계좌 번호",
    "customer_id": "고객 번호",
    "acnt_type": "계좌 유형",
    "prd_type_cd": "상품코드",
    "acnt_bgn_dt": "개설일자",
    "acnt_evlu_amt": "평가적립금",
}
KMAP_DC = {
    "ctrt_no": "계약번호",
    "odtp_name": "근무처명",
    "etco_dt": "입사일자",
    "midl_excc_dt": "중간정산일자",
    "sst_join_dt": "제도가입일자",
    "almt_pymt_prca": "부담금납입원금",
    "utlz_pfls_amt": "운용손익금액",
    "evlu_acca_smtl_amt": "평가적립금합계금액",
}


# ==================== JSON Utils ====================
def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    return str(obj)

def to_json_str(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)

def koreanize_dict(d: Optional[Dict[str, Any]], kmap: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if not d: return None
    return {kmap.get(k, k): v for k, v in d.items()}

def koreanize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = payload.get("context") or {}
    cust = ctx.get("customer")
    accts = ctx.get("accounts") or []
    dc = ctx.get("dc_contract")
    return {
        "파라미터": payload.get("params"),
        "컨텍스트": {
            "고객": koreanize_dict(cust, KMAP_CUSTOMER),
            "계좌들": [koreanize_dict(a, {**KMAP_ACCOUNT, "_account_id": "_account_id"}) for a in accts],
            "DC 계약": koreanize_dict(dc, KMAP_DC),
        },
    }


# ==================== Secrets/DB Helpers ====================
def _safe_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

def _safe_schema(schema: Optional[str]) -> str:
    if not schema: return "public"
    schema = schema.strip()
    return schema if re.fullmatch(r"[A-Za-z0-9_]+", schema) else "public"

def _get_pg_conn_str() -> str:
    host = _safe_secret("PG_HOST", "localhost") or "localhost"
    port = _safe_secret("PG_PORT", "5432") or "5432"
    db   = _safe_secret("PG_DB", "postgres") or "postgres"
    user = _safe_secret("PG_USER", "postgres") or "postgres"
    pwd  = _safe_secret("PG_PASSWORD", None)
    if not pwd:
        raise RuntimeError("PG_PASSWORD가 설정되지 않았습니다. (.streamlit/secrets.toml 또는 환경변수)")
    schema = _safe_schema(_safe_secret("PG_SCHEMA", "public"))
    user_q = quote_plus(user); pwd_q = quote_plus(pwd)
    host_q = quote_plus(host); db_q = quote_plus(db)
    query = urlencode({"options": f"-csearch_path={schema}"})
    return f"postgresql+psycopg://{user_q}:{pwd_q}@{host_q}:{port}/{db_q}?{query}"

def _make_engine_with_schema():
    eng = create_engine(_get_pg_conn_str(), pool_pre_ping=True)
    @event.listens_for(eng, "connect")
    def _on_connect(dbapi_connection, connection_record):
        if register_vector is not None:
            try: register_vector(dbapi_connection)
            except Exception: pass
    return eng


# ==================== DB Loaders ====================
@st.cache_data(ttl=60)
def load_customers_from_db() -> pd.DataFrame:
    engine = _make_engine_with_schema()
    sql = text("""SELECT customer_id, customer_name, brth_dt, age_band FROM kis_customers ORDER BY customer_id""")
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn)
    df["brth_dt"] = pd.to_datetime(df["brth_dt"], errors="coerce").dt.date
    df.rename(columns={
        "customer_id":"고객 번호","customer_name":"고객 이름","brth_dt":"생년월일","age_band":"연령대"
    }, inplace=True)
    df["_customer_id"] = df["고객 번호"]
    return df

@st.cache_data(ttl=60)
def load_accounts_from_db(customer_filter: Optional[Any] = None) -> pd.DataFrame:
    engine = _make_engine_with_schema()
    base_sql = """SELECT account_id, customer_id, acnt_type, prd_type_cd, acnt_bgn_dt, acnt_evlu_amt FROM kis_accounts"""
    params: Dict[str, Any] = {}
    if customer_filter is None:
        stmt = text(base_sql + " ORDER BY account_id")
    else:
        if isinstance(customer_filter, (list, tuple, set)):
            stmt = text(base_sql + " WHERE customer_id IN :cids ORDER BY account_id").bindparams(bindparam("cids", expanding=True))
            params["cids"] = list(customer_filter)
        else:
            stmt = text(base_sql + " WHERE customer_id = :cid ORDER BY account_id")
            params["cid"] = customer_filter

    with engine.begin() as conn:
        df = pd.read_sql(stmt, conn, params=params)

    df["acnt_bgn_dt"] = pd.to_datetime(df["acnt_bgn_dt"], errors="coerce").dt.date
    df["acnt_evlu_amt"] = pd.to_numeric(df["acnt_evlu_amt"], errors="coerce")
    df.rename(columns={
        "account_id":"계좌 번호","customer_id":"고객 번호","acnt_type":"계좌 유형",
        "prd_type_cd":"상품코드","acnt_bgn_dt":"개설일자","acnt_evlu_amt":"평가적립금"
    }, inplace=True)
    df["_account_id"] = df["계좌 번호"]
    df["_customer_id"] = df["고객 번호"]
    return df

@st.cache_data(ttl=60)
def load_dc_contracts_from_db(account_filter: Optional[Any] = None) -> pd.DataFrame:
    engine = _make_engine_with_schema()
    schema = _safe_schema(_safe_secret("PG_SCHEMA", "public"))
    base_sql = f"""
      SELECT ctrt_no, odtp_name, etco_dt, midl_excc_dt, sst_join_dt, almt_pymt_prca, utlz_pfls_amt, evlu_acca_smtl_amt
      FROM {schema}.kis_dc_contract
    """
    params: Dict[str, Any] = {}
    if account_filter is None:
        stmt = text(base_sql + " ORDER BY ctrt_no")
    else:
        if isinstance(account_filter, (list, tuple, set)):
            stmt = text(base_sql + " WHERE ctrt_no IN :ids ORDER BY ctrt_no").bindparams(bindparam("ids", expanding=True))
            params["ids"] = list(account_filter)
        else:
            stmt = text(base_sql + " WHERE ctrt_no = :id ORDER BY ctrt_no")
            params["id"] = account_filter

    with engine.begin() as conn:
        df = pd.read_sql(stmt, conn, params=params)

    for col in ["etco_dt","midl_excc_dt","sst_join_dt"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    for col in ["almt_pymt_prca","utlz_pfls_amt","evlu_acca_smtl_amt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.rename(columns={
        "ctrt_no": "계약번호",
        "odtp_name": "근무처명",
        "etco_dt": "입사일자",
        "midl_excc_dt": "중간정산일자",
        "sst_join_dt": "제도가입일자",
        "almt_pymt_prca": "부담금납입원금",
        "utlz_pfls_amt": "운용손익금액",
        "evlu_acca_smtl_amt": "평가적립금합계금액",
    }, inplace=True)
    df["_ctrt_no"] = df["계약번호"]
    return df


# ==================== AgGrid Helper ====================
def aggrid_table(df: pd.DataFrame, key: str, selection_mode="single", height=280,
                 use_checkbox=True, enable_filter=True, show_side_bar=False):
    gob = GridOptionsBuilder.from_dataframe(df)
    gob.configure_default_column(sortable=True, resizable=True, filter=enable_filter, floatingFilter=False)
    gob.configure_selection(selection_mode=selection_mode, use_checkbox=use_checkbox)
    gob.configure_pagination(enabled=True, paginationPageSize=10)
    if show_side_bar:
        try: gob.configure_side_bar()
        except Exception: pass
    grid_options = gob.build()
    update_mode = GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.MODEL_CHANGED
    return AgGrid(
        df, gridOptions=grid_options, update_mode=update_mode, height=height, key=key,
        fit_columns_on_grid_load=True, allow_unsafe_jscode=True, enable_enterprise_modules=False,
    )

def get_first_value_from_selection(selection, key: str):
    if selection is None: return None
    if isinstance(selection, list):
        if not selection: return None
        first = selection[0]
        return first.get(key) if isinstance(first, dict) else None
    if isinstance(selection, pd.DataFrame):
        if selection.empty or key not in selection.columns: return None
        return selection.iloc[0][key]
    return None

def get_all_values_from_selection(selection, key: str):
    if selection is None: return []
    if isinstance(selection, list):
        return [row.get(key) for row in selection if isinstance(row, dict) and key in row]
    if isinstance(selection, pd.DataFrame) and key in selection.columns:
        return selection[key].dropna().tolist()
    return []


# ==================== Context Builders ====================
def build_context_from_selection() -> Dict[str, Any]:
    selected_customer: Optional[str] = st.session_state.get("selected_customer")
    selected_accounts: List[str] = st.session_state.get("selected_accounts", [])

    cust = None
    if selected_customer:
        row = st.session_state.df_cust.query("_customer_id == @selected_customer")
        if not row.empty:
            r = row.iloc[0]
            cust = {
                "customer_id": r["_customer_id"],
                "customer_name": r["고객 이름"],
                "brth_dt": str(r["생년월일"]),
                "age_band": r["연령대"],
            }

    accts: List[Dict[str, Any]] = []
    if selected_accounts:
        rows = st.session_state.df_acct.query("_account_id in @selected_accounts")
        for _, r in rows.iterrows():
            accts.append({
                "account_id": r["_account_id"],
                "customer_id": r["_customer_id"],
                "acnt_type": r["계좌 유형"],
                "prd_type_cd": r["상품코드"],
                "acnt_bgn_dt": str(r["개설일자"]),
                "acnt_evlu_amt": int(pd.to_numeric(r["평가적립금"], errors="coerce") or 0),
            })

    # DC 계약: 모든 선택 계좌에 대해 매핑(가능한 것만)
    dc = None
    if accts:
        # 첫 번째 DC 계좌 기준 상세
        dc_rows = st.session_state.df_dc.copy() if st.session_state.df_dc is not None else pd.DataFrame()
        if not dc_rows.empty:
            # 첫 행을 대표로 사용
            r = dc_rows.iloc[0]
            dc = {
                "ctrt_no": r["_ctrt_no"],
                "odtp_name": r["근무처명"],
                "etco_dt": str(r["입사일자"]),
                "midl_excc_dt": str(r["중간정산일자"]) if pd.notna(r["중간정산일자"]) else None,
                "sst_join_dt": str(r["제도가입일자"]),
                "almt_pymt_prca": int(pd.to_numeric(r["부담금납입원금"], errors="coerce") or 0),
                "utlz_pfls_amt": int(pd.to_numeric(r["운용손익금액"], errors="coerce") or 0),
                "evlu_acca_smtl_amt": int(pd.to_numeric(r["평가적립금합계금액"], errors="coerce") or 0),
            }

    return {"customer": cust, "accounts": accts, "dc_contract": dc}

def build_context_for_chat() -> Dict[str, Any]:
    return st.session_state.get("context", {"customer": None, "accounts": [], "dc_contract": None})


# ==================== Dummy Simulator / Agent ====================
@tool
def run_pension_simulator(params: dict) -> dict:
    return {
        "source": "dummy",
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "echo_params": params,
        "message": "샘플 더미 응답입니다. 실제 계산 로직으로 교체하세요.",
    }

def make_knowledge_base() -> AgentKnowledge:
    table = os.getenv("AGNO_KG_TABLE", "pension_knowledge")
    search = (os.getenv("AGNO_KG_SEARCH", "hybrid") or "hybrid").lower()
    search_type = SearchType.hybrid if search == "hybrid" else (SearchType.fulltext if search == "fulltext" else SearchType.vector)
    engine = _make_engine_with_schema()
    embedder = OllamaEmbedder(id="openhermes")
    vector_db = PgVector(db_engine=engine, table_name=table, embedder=embedder, search_type=search_type)
    class VectorOnlyKnowledge(AgentKnowledge):
        def __init__(self, vector_db, filters=None, name: str = "pension_knowledge"):
            super().__init__(vector_db=vector_db, filters=filters, name=name)
        @property
        def document_lists(self): return []
        def load(self): return self
    return VectorOnlyKnowledge(vector_db=vector_db, name=table)

def make_agent() -> Agent:
    sys = "당신은 퇴직연금 상담 어시스턴트입니다. 좌측 컨텍스트와 내부 지식만 사용해 답하세요."
    model = Ollama(id="qwen3-h2o2-30b", request_params={"think": False, "keep_alive": "2h"})
    return Agent(system_message=sys, model=model, tools=[run_pension_simulator],
                 markdown=True, knowledge=make_knowledge_base(), search_knowledge=True,
                 enable_agentic_knowledge_filters=True, show_tool_calls=True, debug_mode=False)

AGENT = make_agent()

def context_as_text(ctx: Dict[str, Any]) -> str:
    return "### 선택컨텍스트\n" + to_json_str(ctx)

def run_agent_stream(user_text: str, ctx: Dict[str, Any], debug: bool = False):
    full_prompt = f"{user_text}\n\n{context_as_text(ctx)}"
    st.session_state.last_debug = {"events": [], "error": None, "timing": {}}
    t0 = time.perf_counter()
    try: AGENT.debug_mode = bool(debug)
    except Exception: pass
    try:
        for ev in AGENT.run(full_prompt, stream=True):
            content = getattr(ev, "content", None)
            event_name = getattr(ev, "event", "")
            if content and (event_name == "RunResponseContent" or isinstance(content, str)):
                yield content
        st.session_state.last_debug["timing"]["total_sec"] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        st.session_state.last_debug["error"] = {"message": str(e), "traceback": traceback.format_exc()}
        yield f"\n\n[에러] {e}"


# ==================== Session Init ====================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_debug", {"events": [], "error": None, "timing": {}})
st.session_state.setdefault("sim_params", DEFAULT_PARAM_SCHEMA.copy())
st.session_state.setdefault("context", {"customer": None, "accounts": [], "dc_contract": None})
st.session_state.setdefault("selected_customer", None)
st.session_state.setdefault("selected_accounts", [])

# DB 로드
if "df_cust" not in st.session_state:
    st.session_state.df_cust = load_customers_from_db()
if "df_acct" not in st.session_state:
    st.session_state.df_acct = load_accounts_from_db()
st.session_state.df_dc = load_dc_contracts_from_db(st.session_state.get("selected_accounts") or None)


# ==================== Layout ====================
left, midsep, right = st.columns([0.46, 0.02, 0.52])

# -------- LEFT (고객/계좌/DC) --------
with left:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("고객/계좌 정보")

    # (1) 고객명 검색 + 단건 요약 카드 + 파이차트
    st.caption("① 고객 검색 → 선택")
    all_names = st.session_state.df_cust["고객 이름"].dropna().astype(str).tolist()
    name_selected = st.selectbox("고객 이름 검색", options=[""] + sorted(all_names), index=0,
                                 help="검색창에 일부만 입력해도 됩니다.")

    if name_selected:
        filtered_cust = st.session_state.df_cust[st.session_state.df_cust["고객 이름"] == name_selected]
    else:
        filtered_cust = st.session_state.df_cust.iloc[0:0]  # 선택 없을 때는 비표시

    colL, colR = st.columns([1, 1])
    with colL:
        # 단건 요약(선택 시)
        if not filtered_cust.empty:
            r = filtered_cust.iloc[0]
            st.markdown('<div class="kv">', unsafe_allow_html=True)
            st.markdown(f'<div class="item"><span class="k">고객 번호</span>{r["고객 번호"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="item"><span class="k">고객 이름</span>{r["고객 이름"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="item"><span class="k">생년월일</span>{r["생년월일"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="item"><span class="k">연령대</span>{r["연령대"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 내부 선택 상태 반영
            st.session_state.selected_customer = r["_customer_id"]
        else:
            st.info("고객을 선택하세요.")

    with colR:
        # 파이차트 (고객 선택 시만)
        if not filtered_cust.empty:
            acct_for_pie = load_accounts_from_db(filtered_cust.iloc[0]["고객 번호"])
            if acct_for_pie.empty or pd.to_numeric(acct_for_pie["평가적립금"], errors="coerce").fillna(0).sum() == 0:
                st.info("표시할 평가적립금이 없습니다.")
            else:
                tmp = acct_for_pie.copy()
                tmp["평가적립금"] = pd.to_numeric(tmp["평가적립금"], errors="coerce").fillna(0)
                grp = tmp.groupby("계좌 유형", dropna=False)["평가적립금"].sum().reset_index().sort_values("평가적립금", ascending=False)
                fig = px.pie(grp, names="계좌 유형", values="평가적립금", hole=0.45)
                # 공간 효율: 내부 라벨, 범례 숨김
                fig.update_traces(textinfo="percent+label", textposition="inside",
                                  hovertemplate="%{label}<br>%{value:,}원<br>%{percent}")
                fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig)
        else:
            st.empty()

    st.markdown("---")

    # (2) 계좌 정보: 선택 고객의 모든 계좌
    st.caption("② 선택 고객의 모든 계좌")
    if st.session_state.selected_customer:
        st.session_state.df_acct = load_accounts_from_db(st.session_state.selected_customer)
        df_acct = st.session_state.df_acct
        grid_acct = aggrid_table(
            df_acct[["계좌 번호","고객 번호","계좌 유형","상품코드","개설일자","평가적립금","_account_id","_customer_id"]],
            key=GRID_KEYS["acct"], selection_mode="multiple", height=300, enable_filter=True
        )
        # 컨텍스트: 모든 계좌를 자동 반영
        st.session_state.selected_accounts = df_acct["_account_id"].dropna().tolist()
    else:
        df_acct = pd.DataFrame(columns=["계좌 번호","고객 번호","계좌 유형","상품코드","개설일자","평가적립금","_account_id","_customer_id"])
        st.info("고객 선택 시 계좌가 표시됩니다.")
        st.session_state.selected_accounts = []

    # (3) DC 계약 (선택 계좌 기준)
    st.markdown("---")
    st.caption("③ DC 계약 (계약번호=계좌번호 연결)")
    acct_ids = st.session_state.get("selected_accounts", [])
    st.session_state.df_dc = load_dc_contracts_from_db(acct_ids if acct_ids else None)
    df_dc = st.session_state.df_dc
    if df_dc is not None and not df_dc.empty:
        view_cols = ["계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액","_ctrt_no"]
        use_cols = [c for c in view_cols if c in df_dc.columns]
        grid_dc = aggrid_table(
            df_dc[use_cols], key=GRID_KEYS["dc"], selection_mode="single", height=220, enable_filter=True
        )
    else:
        st.info("표시할 DC 계약 데이터가 없습니다.")

    # ---- 왼쪽 변경 → 컨텍스트 자동 동기화 ----
    new_ctx = build_context_from_selection()
    # 변경 감지(딕셔너리 비교): 같으면 갱신/rerun 안 함
    if to_json_str(new_ctx) != to_json_str(st.session_state.get("context", {})):
        st.session_state.context = new_ctx
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# -------- MID SEP --------
with midsep:
    st.markdown('<div class="v-sep"></div>', unsafe_allow_html=True)


# -------- RIGHT --------
with right:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("챗봇 · 시뮬레이션")

    debug_on = st.toggle("디버그 모드", value=False, help="툴/RAG/이벤트/예외 로그 표시")

    # ---- 파라미터 빌더 ----
    with st.expander("연금 시뮬레이션 파라미터 빌더", expanded=True):
        # 좌: 동작/수동조정, 우: JSON 미리보기(편집 모드 지원)
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### 동작")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear", use_container_width=True, help="컨텍스트를 완전히 비웁니다."):
                    st.session_state.context = {"customer": None, "accounts": [], "dc_contract": None}
                    st.rerun()
            with c2:
                if st.button("Reset", use_container_width=True, help="왼쪽 선택 기준으로 컨텍스트를 다시 세팅합니다."):
                    st.session_state.context = build_context_from_selection()
                    st.rerun()

            st.markdown("#### 수동 조정")
            p = st.session_state.setdefault("sim_params", DEFAULT_PARAM_SCHEMA.copy())
            p["notes"] = st.text_area("메모(선택)", value=p.get("notes") or "", height=100)

            # JSON 복사 버튼 (동작 섹션 쪽)
            payload_preview_left = {"params": st.session_state.sim_params, "context": st.session_state.context}
            show_korean_left = st.checkbox("표시용(한글 라벨)로 보기", value=True, key="show_kor_preview_left")
            display_payload_left = koreanize_payload(payload_preview_left) if show_korean_left else payload_preview_left
            json_str_left = to_json_str(display_payload_left)
            _json_for_js = json_str_left.replace("\\", "\\\\").replace("`", "\\`")
            components.v1.html(
                f"""
                <button id="copy-json-btn" style="padding:.5rem .75rem;cursor:pointer;">📋 JSON 복사</button>
                <script>
                  const btn = document.getElementById('copy-json-btn');
                  btn.addEventListener('click', async () => {{
                    try {{
                      await navigator.clipboard.writeText(`{_json_for_js}`);
                      alert('JSON이 클립보드에 복사되었습니다.');
                    }} catch (e) {{
                      alert('복사에 실패했습니다. 콘솔을 확인하세요.');
                      console.error(e);
                    }}
                  }});
                </script>
                """,
                height=48,
            )

        with col_right:
            st.markdown("#### JSON 미리보기")
            edit_mode = st.toggle("편집 모드(고급): 내부 스키마 JSON 직접 수정", value=False, help="영문 키 기준 payload를 직접 편집합니다.")
            payload_preview = {"params": st.session_state.sim_params, "context": st.session_state.context}

            if not edit_mode:
                show_korean = st.checkbox("표시용(한글 라벨)로 보기", value=True, key="show_kor_preview_right")
                display_payload = koreanize_payload(payload_preview) if show_korean else payload_preview
                st.json(display_payload)
            else:
                raw_str = st.text_area("편집(JSON, 영문 키)", value=to_json_str(payload_preview), height=260)
                apply_col1, apply_col2 = st.columns([1, 3])
                with apply_col1:
                    if st.button("적용", type="primary"):
                        try:
                            new_payload = json.loads(raw_str)
                            if not isinstance(new_payload, dict):
                                raise ValueError("payload는 object여야 합니다.")
                            params = new_payload.get("params", {})
                            context = new_payload.get("context", {})
                            if not isinstance(params, dict) or not isinstance(context, dict):
                                raise ValueError("params/context는 object여야 합니다.")
                            st.session_state.sim_params = {"notes": params.get("notes", "")}
                            cust = context.get("customer")
                            accts = context.get("accounts", [])
                            dc = context.get("dc_contract")
                            if cust is not None and not isinstance(cust, dict):
                                raise ValueError("context.customer는 object여야 합니다.")
                            if not isinstance(accts, list):
                                raise ValueError("context.accounts는 array여야 합니다.")
                            if dc is not None and not isinstance(dc, dict):
                                raise ValueError("context.dc_contract는 object여야 합니다.")
                            st.session_state.context = {
                                "customer": cust or None,
                                "accounts": accts or [],
                                "dc_contract": dc or None,
                            }
                            st.success("JSON을 적용했습니다.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"적용 실패: {e}")
                with apply_col2:
                    st.caption("주의: 내부 키(영문) 스키마로만 편집 가능합니다. UI 표시용 한글 라벨과 다릅니다.")

    st.divider()

    # ---------- 채팅 ----------
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        st.chat_message(role).markdown(msg["content"])

    queued = st.session_state.pop("queued_user_input", None)
    queued_ctx = st.session_state.pop("queued_context", None)
    if queued:
        st.session_state.messages.append({"role": "user", "content": queued})
        st.chat_message("user").markdown(queued)
        ctx = queued_ctx or st.session_state.context
        resp_area = st.chat_message("assistant")
        placeholder = resp_area.empty()
        streamed = ""
        preview = f"(컨텍스트) 고객:{bool(ctx.get('customer'))} / 계좌:{len(ctx.get('accounts', []))} / DC계약:{bool(ctx.get('dc_contract'))}"
        for token in f"질문: {queued}\n\n{preview}".split():
            streamed += token + " "
            placeholder.markdown(streamed)
        st.session_state.messages.append({"role": "assistant", "content": streamed})

    st.markdown("---")
    user_input = st.chat_input("질문을 입력하세요. (예: 현재 컨텍스트 기반으로 DC 관련 규정 설명)")
    if user_input:
        st.session_state["queued_user_input"] = user_input
        st.session_state["queued_context"] = st.session_state.context
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
