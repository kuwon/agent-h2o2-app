# st_sample.py
# -*- coding: utf-8 -*-
# 퇴직연금 RAG + 시뮬레이터 (요청 반영: 2025-09-04)
# - 파라미터 빌더 상단 3버튼 → Clear / Reset (최근입금→월납입 추정 삭제)
# - 수동조정: 버튼 옆 칼럼에 세로 배열, 항목은 notes만 유지
# - 시뮬레이션 유형 제거
# - 하단 초기화/복사용 JSON/전송 버튼 제거, 대신 간결한 복사·다운로드 제공
# - run_pension_simulator: dummy

import os
import re
import json
import time
import traceback
import inspect
from typing import Any, Dict, Optional, List
from urllib.parse import quote_plus, urlencode
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import components
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid, GridUpdateMode
import plotly.express as px

from sqlalchemy.sql import bindparam
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, text, event
import pgvector.sqlalchemy

# (선택) psycopg3용 벡터 등록기 — 리스트 임베딩 경로 안정화용
try:
    from pgvector.psycopg import register_vector
except Exception:
    register_vector = None

# agno / RAG
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAIChat
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
</style>
<div class="center-title">한투 퇴직마스터</div>
""", unsafe_allow_html=True)


# ==================== Constants ====================
# 요청에 따라 notes만 유지
DEFAULT_PARAM_SCHEMA: Dict[str, Any] = {"notes": ""}

GRID_KEYS = {
    "demo": "grid_demo_pg_v3",
    "acct": "grid_acct_pg_v3",
    "dc_ctrt": "grid_dc_contracts",
}

# 한/영 라벨 맵 (표시는 한글, 내부키는 영문 유지)
KMAP_CUSTOMER = {
    "customer_id": "고객 번호",
    "customer_name": "고객 이름",
    "birth": "생년월일",
    "age_band": "연령대",
}
KMAP_ACCOUNT = {
    "account_id": "계좌 번호",
    "customer_id": "고객 번호",
    "product_type": "계좌 유형",
    "prod_code": "상품코드",
    "opened_at": "개설일자",
    "evlu_acca_smtl_amt": "평가적립금",
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


# ==================== Helpers ====================
def _to_py_scalar(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

def _to_py_list(vals):
    return [_to_py_scalar(x) for x in vals]

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
        raise RuntimeError("PG_PASSWORD가 설정되지 않았습니다. .streamlit/secrets.toml 또는 환경변수에 설정해주세요.")
    schema = _safe_schema(_safe_secret("PG_SCHEMA", "public"))
    user_q = quote_plus(user); pwd_q = quote_plus(pwd)
    host_q = quote_plus(host); db_q = quote_plus(db)
    query = urlencode({"options": f"-csearch_path={schema}"})
    # psycopg3 사용
    return f"postgresql+psycopg://{user_q}:{pwd_q}@{host_q}:{port}/{db_q}?{query}"

def _make_engine_with_schema():
    eng = create_engine(_get_pg_conn_str(), pool_pre_ping=True)
    # 벡터 등록(보조책)
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
    if "brth_dt" in df.columns:
        df["brth_dt"] = pd.to_datetime(df["brth_dt"], errors="coerce").dt.date
    df.rename(columns={"customer_id":"고객 번호","customer_name":"고객 이름","brth_dt":"생년월일","age_band":"연령대"}, inplace=True)
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
            params["cids"] = _to_py_list(list(customer_filter))
        else:
            stmt = text(base_sql + " WHERE customer_id = :cid ORDER BY account_id")
            params["cid"] = _to_py_scalar(customer_filter)
    with engine.begin() as conn:
        df = pd.read_sql(stmt, conn, params=params)
    if "acnt_bgn_dt" in df.columns:
        df["acnt_bgn_dt"] = pd.to_datetime(df["acnt_bgn_dt"], errors="coerce").dt.date
    if "acnt_evlu_amt" in df.columns:
        df["acnt_evlu_amt"] = pd.to_numeric(df["acnt_evlu_amt"], errors="coerce")
    df.rename(columns={
        "account_id":"계좌 번호","customer_id":"고객 번호","acnt_type":"계좌 유형",
        "prd_type_cd":"상품코드","acnt_bgn_dt":"개설일자","acnt_evlu_amt":"평가적립금"
    }, inplace=True)
    df["_account_id"] = df["계좌 번호"]
    df["_customer_id"] = df["고객 번호"]
    return df

@st.cache_data(ttl=60)
def load_dc_contracts_from_db(account_filter=None) -> pd.DataFrame:
    """
    kis_dc_contract: ctrt_no(계약번호) = kis_accounts.account_id 와 연결
    """
    engine = _make_engine_with_schema()
    schema = _safe_schema(_safe_secret("PG_SCHEMA", "public"))
    base_sql = f"""
        SELECT ctrt_no, odtp_name, etco_dt, midl_excc_dt, sst_join_dt,
               almt_pymt_prca, utlz_pfls_amt, evlu_acca_smtl_amt
        FROM {schema}.kis_dc_contract
    """
    params = {}
    if account_filter is None:
        stmt = text(base_sql + " ORDER BY ctrt_no")
    else:
        if isinstance(account_filter, (list, tuple, set)):
            ids = _to_py_list(list(account_filter))
            if not ids:
                return pd.DataFrame(columns=[
                    "계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액","_ctrt_no"
                ])
            stmt = text(base_sql + " WHERE ctrt_no IN :ids ORDER BY ctrt_no").bindparams(bindparam("ids", expanding=True))
            params["ids"] = ids
        else:
            stmt = text(base_sql + " WHERE ctrt_no = :id ORDER BY ctrt_no")
            params["id"] = _to_py_scalar(account_filter)
    with engine.begin() as conn:
        df = pd.read_sql(stmt, conn, params=params)

    # 정리/한글 컬럼
    for col in ["etco_dt","midl_excc_dt","sst_join_dt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    for col in ["almt_pymt_prca","utlz_pfls_amt","evlu_acca_smtl_amt"]:
        if col in df.columns:
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
    df["_ctrt_no"] = df["계약번호"] if "계약번호" in df.columns else None
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
        vals = [row.get(key) for row in selection if isinstance(row, dict) and key in row]
        return [_to_py_scalar(v) for v in vals]
    if isinstance(selection, pd.DataFrame):
        if key in selection.columns:
            return [_to_py_scalar(v) for v in selection[key].dropna().tolist()]
    return []


# ==================== Context Builders ====================
def build_context_from_selection() -> Dict[str, Any]:
    # 좌측 선택 기반으로 컨텍스트 구성
    selected_customer: Optional[str] = st.session_state.get("selected_customer")
    selected_accounts: List[str] = st.session_state.get("selected_accounts", [])

    cust = None
    if selected_customer:
        row = st.session_state.demo_df.query("_customer_id == @selected_customer")
        if not row.empty:
            r = row.iloc[0]
            cust = {
                "customer_id": r["_customer_id"],
                "customer_name": r["고객 이름"],
                "birth": str(r["생년월일"]),
                "age_band": r["연령대"],
            }

    accts: List[Dict[str, Any]] = []
    if selected_accounts:
        rows = st.session_state.acct_df.query("_account_id in @selected_accounts")
        for _, r in rows.iterrows():
            accts.append({
                "account_id": r["_account_id"],
                "customer_id": r["_customer_id"],
                "product_type": r["계좌 유형"],
                "prod_code": r["상품코드"],
                "opened_at": str(r["개설일자"]),
                "evlu_acca_smtl_amt": int(_to_py_scalar(r["평가적립금"])),
            })

    # DC 계약: 첫 번째 DC 계좌를 기준으로 연결 (계약번호 = account_id)
    dc = None
    if accts:
        dc_candidates = [a for a in accts if a["product_type"] == "DC"]
        if dc_candidates:
            aid = dc_candidates[0]["account_id"]
            row = st.session_state.dc_df.query("_ctrt_no == @aid")
            if not row.empty:
                r = row.iloc[0]
                dc = {
                    "ctrt_no": r["_ctrt_no"],
                    "odtp_name": r["근무처명"],
                    "etco_dt": str(r["입사일자"]),
                    "midl_excc_dt": str(r["중간정산일자"]) if pd.notna(r["중간정산일자"]) else None,
                    "sst_join_dt": str(r["제도가입일자"]),
                    "almt_pymt_prca": int(_to_py_scalar(r["부담금납입원금"])),
                    "utlz_pfls_amt": int(_to_py_scalar(r["운용손익금액"])),
                    "evlu_acca_smtl_amt": int(_to_py_scalar(r["평가적립금합계금액"])),
                }

    return {"customer": cust, "accounts": accts, "dc_contract": dc}

def build_context_for_chat() -> Dict[str, Any]:
    # 현재 세션의 context를 그대로 사용
    return st.session_state.get("context", {"customer": None, "accounts": [], "dc_contract": None})


# ==================== Tools / Agent ====================
@tool
def run_pension_simulator(params: dict) -> dict:
    """요청에 따라 더미 버전"""
    return {
        "source": "dummy",
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "echo_params": params,
        "message": "샘플 더미 응답입니다. 실제 계산 로직/서비스 연동으로 교체하세요.",
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
    sys = (
        "당신은 퇴직연금 상담 어시스턴트입니다. "
        "웹 검색 없이 pgvector(pension_knowledge)와 좌측 컨텍스트만 사용해 답하세요. "
        "필요 시 run_pension_simulator 툴을 사용할 수 있습니다."
    )
    model = OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    model = Ollama(id="qwen3-h2o2-30b", request_params={"think": False, "keep_alive": "2h"})
    agent = Agent(
        system_message=sys,
        model=model,
        tools=[run_pension_simulator],
        markdown=True,
        knowledge=make_knowledge_base(),
        search_knowledge=True,
        enable_agentic_knowledge_filters=True,
        show_tool_calls=True,
        debug_mode=False,
    )
    return agent

AGENT = make_agent()

def context_as_text(ctx: Dict[str, Any]) -> str:
    return "### 선택컨텍스트\n" + to_json_str(ctx)

def run_agent_stream(user_text: str, ctx: Dict[str, Any], debug: bool = False):
    full_prompt = f"{user_text}\n\n{context_as_text(ctx)}"
    st.session_state.last_debug = {"events": [], "error": None, "timing": {}}
    t0 = time.perf_counter()
    try:
        AGENT.debug_mode = bool(debug)
    except Exception:
        pass
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

# Data load
if "demo_df" not in st.session_state:
    st.session_state.demo_df = load_customers_from_db()
if "acct_df" not in st.session_state:
    st.session_state.acct_df = load_accounts_from_db()
st.session_state.setdefault("selected_customer", None)
st.session_state.setdefault("selected_accounts", [])
# DC 캐시(선택 계좌 기준으로 갱신)
st.session_state.dc_df = load_dc_contracts_from_db(st.session_state.get("selected_accounts") or None)


# ==================== Layout ====================
left, midsep, right = st.columns([0.46, 0.02, 0.52])

# -------- LEFT --------
with left:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("고객/계좌 정보")

    # 고객 그리드
    st.caption("고객을 하나 선택하세요 (싱글 선택)")
    grid = aggrid_table(
        st.session_state.demo_df[["고객 번호", "고객 이름", "생년월일", "연령대", "_customer_id"]],
        key=GRID_KEYS["demo"], selection_mode="single", height=260, enable_filter=True
    )
    sel = grid.get("selected_rows", None)
    st.session_state.selected_customer = get_first_value_from_selection(sel, "_customer_id")

    # 계좌그리드용 고객 필터
    visible_ids = None
    try:
        if "data" in grid and isinstance(grid["data"], list) and grid["data"]:
            visible_df = pd.DataFrame(grid["data"])
            if "_customer_id" in visible_df.columns:
                visible_ids = visible_df["_customer_id"].dropna().unique().tolist()
    except Exception:
        visible_ids = None

    st.info(
        f"선택 고객: {st.session_state.selected_customer or '없음'}"
        + (f" | 필터된 고객 수: {len(visible_ids)}" if visible_ids is not None else "")
    )

    st.markdown("---")
    only_selected = st.checkbox("선택 고객의 계좌만 보기", value=True)

    if only_selected:
        if st.session_state.selected_customer is not None:
            current_acct_df = load_accounts_from_db(st.session_state.selected_customer)
        elif visible_ids:
            current_acct_df = load_accounts_from_db(visible_ids)
        else:
            current_acct_df = load_accounts_from_db()
    else:
        current_acct_df = load_accounts_from_db()

    st.session_state.acct_df = current_acct_df

    # 파이 차트
    st.subheader("계좌 유형별 평가적립금")
    pie_df = current_acct_df.copy()
    if pie_df.empty or pie_df["평가적립금"].fillna(0).sum() == 0:
        st.info("표시할 평가적립금이 없습니다. 고객을 선택하거나 계좌를 선택해 주세요.")
    else:
        grp = pie_df.groupby("계좌 유형", dropna=False)["평가적립금"].sum().reset_index().sort_values("평가적립금", ascending=False)
        fig = px.pie(grp, names="계좌 유형", values="평가적립금", hole=0.4)
        fig.update_traces(textinfo="percent+label", textposition="inside", hovertemplate="%{label}<br>%{value:,}원<br>%{percent}")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="계좌 유형")
        st.plotly_chart(fig, use_container_width=True)

    # 계좌 그리드
    st.caption("계좌를 선택하세요 (멀티 선택 가능)")
    grid_acct = aggrid_table(
        current_acct_df[["계좌 번호", "고객 번호", "계좌 유형", "상품코드", "개설일자", "평가적립금", "_account_id", "_customer_id"]],
        key=GRID_KEYS["acct"], selection_mode="multiple", height=320, enable_filter=True
    )
    sel_acct = grid_acct.get("selected_rows", None)
    st.session_state.selected_accounts = get_all_values_from_selection(sel_acct, "_account_id")

    # DC 계약 그리드 (선택 계좌 기준)
    st.markdown("### ③ DC 계약")
    acct_ids = st.session_state.get("selected_accounts", [])
    try:
        st.session_state.dc_df = load_dc_contracts_from_db(acct_ids if acct_ids else None)
    except Exception:
        st.session_state.dc_df = pd.DataFrame()

    dc_df = st.session_state.dc_df
    view_cols = ["계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액","_ctrt_no"]
    cols = [c for c in view_cols if c in getattr(dc_df, "columns", [])]
    if dc_df is not None and not dc_df.empty and cols:
        st.caption("DC 계약을 선택하세요")
        grid_dc = aggrid_table(dc_df[cols], key=GRID_KEYS["dc_ctrt"], selection_mode="single", height=260, enable_filter=True)
        sel_dc = grid_dc.get("selected_rows", None)
        st.session_state["selected_dc_row"] = sel_dc[0] if isinstance(sel_dc, list) and len(sel_dc) > 0 else None
    else:
        st.info("표시할 DC 계약 데이터가 없습니다.")
        st.session_state["selected_dc_row"] = None

    with st.expander("선택 컨텍스트 미리보기", expanded=False):
        # 현재 세션 컨텍스트를 한글 표시로 미리보기
        preview_ctx = st.session_state.get("context", {"customer": None, "accounts": [], "dc_contract": None})
        disp = {
            "customer": {KMAP_CUSTOMER.get(k,k): v for k,v in (preview_ctx.get("customer") or {}).items()},
            "accounts": [{KMAP_ACCOUNT.get(k,k): v for k,v in a.items()} for a in (preview_ctx.get("accounts") or [])],
            "dc_contract": {KMAP_DC.get(k,k): v for k,v in (preview_ctx.get("dc_contract") or {}).items()} if preview_ctx.get("dc_contract") else None,
        }
        st.json(disp)

    st.markdown('</div>', unsafe_allow_html=True)


# -------- MID SEP --------
with midsep:
    st.markdown('<div class="v-sep"></div>', unsafe_allow_html=True)


# -------- RIGHT --------
with right:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("챗봇 · 시뮬레이션")

    # 디버그 & KB 헬스
    debug_on = st.toggle("디버그 모드", value=False, help="툴/RAG/이벤트/예외 로그 표시")

    # ---- 파라미터 빌더 ----
    with st.expander("연금 시뮬레이션 파라미터 빌더", expanded=True):
        col_btns, col_params = st.columns([1, 1], gap="large")

        with col_btns:
            st.markdown("#### 동작")
            c1, c2 = st.columns(2)
            # Clear: 컨텍스트 완전 비우기
            with c1:
                if st.button("Clear", use_container_width=True, help="컨텍스트를 완전히 비웁니다."):
                    st.session_state.context = {"customer": None, "accounts": [], "dc_contract": None}
                    st.success("컨텍스트를 비웠습니다.")
            # Reset: 왼쪽 선택 기준으로 컨텍스트 재세팅
            with c2:
                if st.button("Reset", use_container_width=True, help="왼쪽 선택 기준으로 컨텍스트를 다시 세팅합니다."):
                    st.session_state.context = build_context_from_selection()
                    st.success("왼쪽 선택으로 컨텍스트를 재세팅했습니다.")

        with col_params:
            st.markdown("#### 수동 조정")
            p = st.session_state.sim_params
            # 요청: 세로 배열 + notes만 유지
            p["notes"] = st.text_area("메모(선택)", value=p.get("notes") or "", height=100)

        # payload 미리보기 (한글 라벨 사본 표시는 선택)
        payload_preview = {"params": st.session_state.sim_params, "context": build_context_for_chat()}
        st.markdown("#### JSON 미리보기")
        show_korean = st.checkbox("표시용(한글 라벨)로 보기", value=True)
        if show_korean:
            disp = {
                "params": payload_preview["params"],
                "context": {
                    "고객": {KMAP_CUSTOMER.get(k,k): v for k,v in (payload_preview["context"].get("customer") or {}).items()},
                    "계좌들": [{KMAP_ACCOUNT.get(k,k): v for k,v in a.items()} for a in (payload_preview["context"].get("accounts") or [])],
                    "DC 계약": {KMAP_DC.get(k,k): v for k,v in (payload_preview["context"].get("dc_contract") or {}).items()} if payload_preview["context"].get("dc_contract") else None,
                }
            }
            st.json(disp)
            json_str = to_json_str(disp)
        else:
            st.json(payload_preview)
            json_str = to_json_str(payload_preview)

        # 간결한 복사/다운로드
        _json_for_js = json_str.replace("\\", "\\\\").replace("`", "\\`")
        col_copy, col_dl, _ = st.columns([1, 1, 6])
        with col_copy:
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
        with col_dl:
            st.download_button(
                "⬇ JSON 다운로드",
                data=json_str.encode("utf-8"),
                file_name="pension_payload.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        st.caption("※ 채팅은 좌측 컨텍스트를 자동 사용합니다. 전송 버튼 없이 아래 입력창에서 바로 질문하세요.")

    st.divider()

    # ---------- 채팅 ----------
    # 히스토리 렌더
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        st.chat_message(role).markdown(msg["content"])

    # 큐 처리 (이전 프레임에서 입력된 것)
    queued = st.session_state.pop("queued_user_input", None)
    queued_ctx = st.session_state.pop("queued_context", None)
    if queued:
        st.session_state.messages.append({"role": "user", "content": queued})
        st.chat_message("user").markdown(queued)
        ctx = queued_ctx or build_context_for_chat()
        resp_area = st.chat_message("assistant")
        placeholder = resp_area.empty()
        streamed = ""
        for chunk in run_agent_stream(queued, ctx, debug=debug_on):
            streamed += chunk
            placeholder.markdown(streamed)
        st.session_state.messages.append({"role": "assistant", "content": streamed})

    # 입력창은 항상 맨 아래
    st.markdown("---")
    user_input = st.chat_input("질문을 입력하세요. (예: 현재 컨텍스트 기반으로 DC 관련 규정 설명)")
    if user_input:
        st.session_state["queued_user_input"] = user_input
        st.session_state["queued_context"] = build_context_for_chat()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
