# st_sample.py (v15: chat above input + scroll after input, context builder visible, new ollama defaults/options)
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import traceback
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import quote_plus, urlencode
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import components
import plotly.express as px

# AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid

# SQLAlchemy / pgvector
from sqlalchemy.sql import bindparam
from sqlalchemy import create_engine, text, event
import pgvector.sqlalchemy
try:
    from pgvector.psycopg import register_vector
except Exception:
    register_vector = None

# agno
from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
try:
    from agno.models.openai import OpenAIChat
except Exception:
    OpenAIChat = None
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.agent import AgentKnowledge


# ==================== Page Config & Styles ====================
st.set_page_config(page_title="한투 퇴직마스터", layout="wide")
st.markdown("""
<style>
/* tighter global padding */
.main .block-container { padding-top: .2rem; padding-left: .5rem; padding-right: .5rem; }
/* bigger centered title */
.center-title { text-align:center; margin: .25rem 0 .6rem 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -0.02em; }
/* slimmer panels */
.panel-soft { padding: 10px 12px; border-radius: 12px; background: #ffffff;
  border: 1px solid rgba(0,0,0,.06); box-shadow: 0 1px 2px rgba(0,0,0,.03); }
.panel-soft.flush-top { padding-top: 0; }
.panel-soft > :first-child { margin-top: 0 !important; }
/* slimmer vertical separator */
.v-sep { border-left: 1px solid #e9ecef; height: calc(100vh - 140px); margin: 6px 4px; }
/* data summary */
.summary-card { border:1px solid #e9ecef; border-radius:12px; padding:10px 12px; }
.summary-card table { width:100%; font-size:14px; border-collapse:collapse; }
.summary-card td { padding:6px 4px; }
.small-note { color:#6c757d; font-size:12px; margin-top:4px; }
/* lightweight badge for hidden thoughts */
.badge-thinking { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#f1f3f5; color:#495057; margin:0 4px; }
</style>
<div class="center-title">한투 퇴직마스터</div>
""", unsafe_allow_html=True)


# ==================== Constants ====================
DEFAULT_PARAM_SCHEMA: Dict[str, Any] = {"notes": ""}
GRID_KEYS = {"acct": "grid_acct_v15", "dc": "grid_dc_v15"}

# 라벨 맵
KMAP_CUSTOMER = {"customer_id": "고객 번호","customer_name": "고객 이름","brth_dt": "생년월일","age_band": "연령대"}
KMAP_ACCOUNT  = {"account_id": "계좌 번호","customer_id": "고객 번호","acnt_type": "계좌 유형","prd_type_cd": "상품코드","acnt_bgn_dt": "개설일자","acnt_evlu_amt": "평가적립금"}
KMAP_DC       = {"ctrt_no": "계약번호","odtp_name": "근무처명","etco_dt": "입사일자","midl_excc_dt": "중간정산일자","sst_join_dt": "제도가입일자","almt_pymt_prca": "부담금납입원금","utlz_pfls_amt": "운용손익금액","evlu_acca_smtl_amt": "평가적립금합계금액"}


# ==================== JSON Utils ====================
def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    return str(obj)

def to_json_str(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)

def koreanize_dict(d: Optional[Dict[str, Any]], kmap: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if d is None: return None
    if isinstance(d, dict) and len(d) == 0: return {}
    return {kmap.get(k, k): v for k, v in d.items()}

def koreanize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = payload.get("context")
    ctx = ctx if isinstance(ctx, dict) else {}
    cust = ctx.get("customer")
    accts = ctx.get("accounts") or []
    dcs = ctx.get("dc_contracts") or []
    return {
        "파라미터": payload.get("params"),
        "컨텍스트": {} if len(ctx) == 0 else {
            "고객": koreanize_dict(cust, KMAP_CUSTOMER),
            "계좌들": [koreanize_dict(a, {**KMAP_ACCOUNT, "_account_id": "_account_id"}) for a in accts],
            "DC 계약": [koreanize_dict(x, KMAP_DC) for x in dcs],
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
        raise RuntimeError("PG_PASSWORD가 설정되지 않았습니다.")
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
    df["_account_id"] = df["계좌 번호"].astype(str)
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
    if account_filter is None or (isinstance(account_filter, list) and len(account_filter) == 0):
        return pd.DataFrame(columns=["계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액","_ctrt_no"])
    else:
        if isinstance(account_filter, (list, tuple, set)):
            stmt = text(base_sql + " WHERE ctrt_no IN :ids ORDER BY ctrt_no").bindparams(bindparam("ids", expanding=True))
            params["ids"] = [str(x) for x in list(account_filter)]
        else:
            stmt = text(base_sql + " WHERE ctrt_no = :id ORDER BY ctrt_no")
            params["id"] = str(account_filter)

    with engine.begin() as conn:
        df = pd.read_sql(stmt, conn, params=params)

    for col in ["etco_dt","midl_excc_dt","sst_join_dt"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    for col in ["almt_pymt_prca","utlz_pfls_amt","evlu_acca_smtl_amt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.rename(columns={
        "ctrt_no": "계약번호","odtp_name": "근무처명","etco_dt": "입사일자","midl_excc_dt": "중간정산일자",
        "sst_join_dt": "제도가입일자","almt_pymt_prca": "부담금납입원금","utlz_pfls_amt": "운용손익금액","evlu_acca_smtl_amt": "평가적립금합계금액",
    }, inplace=True)
    df["_ctrt_no"] = df["계약번호"].astype(str)
    return df


# ==================== AgGrid Helper ====================
def aggrid_table(
    df: pd.DataFrame, key: str, selection_mode="none", height=280,
    enable_filter=True, fit_columns_on_load=True, allow_horizontal_scroll=False
):
    gob = GridOptionsBuilder.from_dataframe(df)
    gob.configure_default_column(sortable=True, resizable=True, filter=enable_filter, wrapText=False, autoHeight=False)
    if selection_mode in ("single", "multiple"):
        gob.configure_selection(selection_mode=selection_mode, use_checkbox=(selection_mode=="multiple"))
    grid_options = gob.build()
    grid_options["suppressHorizontalScroll"] = not allow_horizontal_scroll

    # Prefer new API: update_on accepts grid events, fall back to deprecated update_mode if needed.
    try:
        return AgGrid(
            df, gridOptions=grid_options, update_on=["filterChanged", "modelUpdated"],
            height=height, key=key,
            fit_columns_on_grid_load=bool(fit_columns_on_load), allow_unsafe_jscode=True, enable_enterprise_modules=False,
        )
    except TypeError:
        # Fallback for older st-aggrid
        from st_aggrid import GridUpdateMode  # lazy import to avoid global deprecation warning
        update_mode = GridUpdateMode.MODEL_CHANGED | GridUpdateMode.FILTERING_CHANGED
        return AgGrid(
            df, gridOptions=grid_options, update_mode=update_mode,
            height=height, key=key,
            fit_columns_on_grid_load=bool(fit_columns_on_load), allow_unsafe_jscode=True, enable_enterprise_modules=False,
        )


# ==================== Context Builders ====================
def _rows_to_accounts(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if rows is None or rows.empty: 
        return out
    for _, r in rows.iterrows():
        out.append({
            "account_id": str(r["_account_id"]),
            "customer_id": r["_customer_id"],
            "acnt_type": r["계좌 유형"],
            "prd_type_cd": r["상품코드"],
            "acnt_bgn_dt": str(r["개설일자"]),
            "acnt_evlu_amt": int(pd.to_numeric(r["평가적립금"], errors="coerce") or 0),
        })
    return out

def _rows_to_dc_list(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if rows is None or rows.empty: return out
    for _, r in rows.iterrows():
        out.append({
            "ctrt_no": str(r["_ctrt_no"]),
            "odtp_name": r["근무처명"],
            "etco_dt": str(r["입사일자"]),
            "midl_excc_dt": str(r["중간정산일자"]) if pd.notna(r["중간정산일자"]) else None,
            "sst_join_dt": str(r["제도가입일자"]),
            "almt_pymt_prca": int(pd.to_numeric(r["부담금납입원금"], errors="coerce") or 0),
            "utlz_pfls_amt": int(pd.to_numeric(r["운용손익금액"], errors="coerce") or 0),
            "evlu_acca_smtl_amt": int(pd.to_numeric(r["평가적립금합계금액"], errors="coerce") or 0),
        })
    return out

def build_context_from_selection() -> Dict[str, Any]:
    selected_customer: Optional[str] = st.session_state.get("selected_customer")
    df_acct: pd.DataFrame = st.session_state.get("df_acct", pd.DataFrame())
    df_dc: pd.DataFrame = st.session_state.get("df_dc", pd.DataFrame())

    cust = None
    if selected_customer:
        row = st.session_state.df_cust.query("_customer_id == @selected_customer")
        if not row.empty:
            r = row.iloc[0]
            cust = {"customer_id": r["_customer_id"], "customer_name": r["고객 이름"],
                    "brth_dt": str(r["생년월일"]), "age_band": r["연령대"]}

    accts = _rows_to_accounts(df_acct)
    dc_list = _rows_to_dc_list(df_dc)

    return {"customer": cust, "accounts": accts, "dc_contracts": dc_list}

def build_context_for_chat() -> Dict[str, Any]:
    ctx = st.session_state.get("context")
    return ctx if isinstance(ctx, dict) else {}


# ==================== Dummy Simulator / Agent Factory ====================
@tool
def run_pension_simulator(params: dict) -> dict:
    return {"source": "dummy","as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),"echo_params": params,"message": "샘플 더미 응답입니다."}

def make_knowledge_base(provider) -> AgentKnowledge:
    env_name = 'AGNO_OPENAI_KG_TABLE' if provider == 'openai' else 'AGNO_OLLAMA_KG_TABLE'

    table = os.getenv(env_name, "kis_pension_knowledge")
    embedder = OpenAIEmbedder() if provider == "openai" else OllamaEmbedder(id="openhermes")

    search = (os.getenv("AGNO_KG_SEARCH", "hybrid") or "hybrid").lower()
    search_type = SearchType.hybrid if search == "hybrid" else (SearchType.fulltext if search == "fulltext" else SearchType.vector)
    engine = _make_engine_with_schema()
    
    vector_db = PgVector(db_engine=engine, table_name=table, embedder=embedder, search_type=search_type)
    class VectorOnlyKnowledge(AgentKnowledge):
        def __init__(self, vector_db, filters=None, name: str = "kis_pension_knowledge"):
            super().__init__(vector_db=vector_db, filters=filters, name=name)
        @property
        def document_lists(self): return []
        def load(self): return self
    return VectorOnlyKnowledge(vector_db=vector_db, name=table)

def make_agent(provider: str, model_id: str, req: Dict[str, Any], search_knowledge: bool) -> Agent:
    sys = """
        당신은 퇴직연금 상담 어시스턴트입니다. 항상 한국어로 응답하며, 아래 원칙을 따릅니다.

        [의도 파악]
        - 질문을 (A) 컨텍스트 질의(고객/계좌/계약 상태·계산), (B) 정책·용어 질의(규정·절차·FAQ), (C) 실행 요청(시뮬레이션·비교) 중 하나로 분류합니다.
        - 혼합일 때는 핵심 의도 1개를 우선 처리하고, 나머지는 “다음 단계”로 제안합니다.

        [정보 출처 우선순위 — RAG 최우선]
        1) Knowledge/RAG(FAQ 포함): 정책·용어·절차 관련 답변은 반드시 RAG 근거를 사용합니다(FAQ 우선).
        2) 컨텍스트(JSON): 고객·계좌·DC 계약(DC는 ctrt_no=account_id 매핑) 조건으로 개인화합니다.
        3) 추정 금지: 컨텍스트/지식에 없으면 부족함을 명시하고, 필요한 경우 보충 질문 1개만 합니다.

        [맞춤형 응답]
        - 컨텍스트를 반영한 조건(계좌유형, 금액, 날짜, 계약)을 명시적으로 표기합니다.
        - 숫자는 1,234,567 형식, 날짜는 YYYY-MM-DD 형식을 권장합니다.

        [답변 형식]
        - 요약(3줄 이내) → 설명/근거(불릿·표) → 유의사항 → 다음 단계 제안.
        - 근거에는 지식 문서의 제목/섹션/개정일만 짧게 표기(내부 ID·경로·로그 노출 금지).
        - 요청 시에만 JSON을 보여주며, 내부 키/ID는 숨깁니다.

        [도구/시뮬레이션]
        - 계산/비교/시나리오가 필요하면 run_pension_simulator 도구를 호출하고 핵심 결과만 요약합니다.

        [불확실성]
        - 아는 사실과 가정을 분리하고, 추가로 필요한 정보 1개만 구체적으로 요청한 후 가능한 범위의 최선 답변을 제시합니다.

        [금지·톤]
        - 생각 과정(COT)·내부 체계 노출 금지. 금융·세법 해석은 일반 안내로 제한, 필요 시 전문가 상담 권고를 짧게 덧붙입니다.
        - 전문적이되 친절·간결한 톤을 유지합니다.

        [권장 출력 프레임]
        - 요약: 2–3문장
        - 고객/계좌 기준 설명: 불릿 2–5개
        - 규정/근거(요약): 불릿 2–4개 (문서명/섹션/개정일)
        - 다음 단계: 1–2개 (예: “중간정산 자격 확인”, “시뮬레이터 실행”)

    """
    if provider == "openai":
        if OpenAIChat is None:
            raise RuntimeError("OpenAIChat 모델이 현재 agno 버전에 없습니다. agno 업데이트가 필요합니다.")
        api_key = _safe_secret("OPENAI_API_KEY", None)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

        # ✅ OpenAI는 max_tokens < 1 허용 안 함 → 안전하게 보정
        opts = req.get("options", {}) or {}
        temperature = float(opts.get("temperature", 0.3))
        top_p = float(opts.get("top_p", 0.9))
        max_tokens_raw = opts.get("num_predict", 1024)
        try:
            max_tokens = int(max_tokens_raw)
        except Exception:
            max_tokens = 1024
        if max_tokens < 1:
            max_tokens = 1024  # 기본값으로 보정

        model = OpenAIChat(
            id=model_id,
            api_key=api_key,
            request_params={
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
        )
    else:
        # Ollama: 모든 세부 옵션은 'options'로 전달
        model = Ollama(id=model_id, request_params=req)

    return Agent(
        system_message=sys,
        model=model,
        tools=[run_pension_simulator],
        markdown=True,
        knowledge=make_knowledge_base(provider),
        search_knowledge=search_knowledge,
        enable_agentic_knowledge_filters=True,
        show_tool_calls=True,
        debug_mode=False,
    )


def _normalize_llm_cfg(raw: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], bool]:
    """provider, model_id, request_params, search_knowledge"""
    provider = raw.get("provider", "ollama")
    model_id = raw.get("model_id", "qwen3-h2o2-14b")
    # 통일된 options dict로 정리
    options = raw.get("options") or {}
    # 기본값 보강
    opts = {
        "temperature": float(options.get("temperature", 0.3)),
        "top_p": float(options.get("top_p", 0.9)),
        "top_k": int(options.get("top_k", 40)),
        "repeat_penalty": float(options.get("repeat_penalty", 1.05)),
        "num_ctx": int(options.get("num_ctx", 16384)),
        "num_predict": int(options.get("num_predict", -1)),
        "num_batch": int(options.get("num_batch", 512)),
        "num_thread": int(options.get("num_thread", 0)),
    }
    keep_alive = raw.get("keep_alive", "2h")
    req = {"keep_alive": keep_alive, "options": opts}
    return provider, model_id, req, bool(raw.get("search_knowledge", True))

def get_agent():
    cfg = st.session_state.get("agent_cfg") or {
        "provider":"ollama","model_id":"qwen3-h2o2-14b",
        "options":{"temperature":0.3,"top_p":0.9,"top_k":40,"repeat_penalty":1.05,"num_ctx":16384,"num_predict":-1,"num_batch":512,"num_thread":0},
        "keep_alive":"2h","search_knowledge":True
    }
    provider, model_id, req, search_knowledge = _normalize_llm_cfg(cfg)
    sig = (provider, model_id, json.dumps(req, sort_keys=True), search_knowledge)
    if "AGENT" not in st.session_state or st.session_state.get("_agent_sig") != sig:
        st.session_state.AGENT = make_agent(provider, model_id, req, search_knowledge)
        st.session_state._agent_sig = sig
    return st.session_state.AGENT

def context_as_text(ctx: Dict[str, Any]) -> str:
    return "### 선택컨텍스트\n" + to_json_str(ctx)

# ====== Think Masking ======
def mask_thoughts(text: str, notice_inserted: bool) -> Tuple[str, bool]:
    t = text

    # 1) XML-style <think>...</think>
    pat_xml = re.compile(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>")
    if pat_xml.search(t):
        if not notice_inserted:
            t = pat_xml.sub(" <span class='badge-thinking'>생각 중…</span> ", t, count=1)
            notice_inserted = True
        t = pat_xml.sub("", t)

    # 2) Fenced code blocks ```think/analysis/...```
    pat_fence = re.compile(r"(?is)```(?:\s*(?:think|thoughts|analysis|chain[_ -]?of[_ -]?thought)[^\n]*)\n.*?```")
    if pat_fence.search(t):
        if not notice_inserted:
            t = pat_fence.sub(" <span class='badge-thinking'>생각 중…</span> ", t, count=1)
            notice_inserted = True
        t = pat_fence.sub("", t)

    # 3) Bracketed tokens [think]...[/think] or 【Thinking】…
    pat_br = re.compile(r"(?is)[\[\{（(【]\s*(?:think|thinking|thoughts)\s*[\]\}）)】].*?[\[\{（(【]\s*/?\s*(?:think|thinking|thoughts)\s*[\]\}）)】]")
    if pat_br.search(t):
        if not notice_inserted:
            t = pat_br.sub(" <span class='badge-thinking'>생각 중…</span> ", t, count=1)
            notice_inserted = True
        t = pat_br.sub("", t)

    # 4) Streaming partial start tokens: "...<think>" without close, or "think:" prefix at top
    lower = t.lower()
    s = lower.rfind("<think>")
    e = lower.rfind("</think>")
    if s != -1 and (e == -1 or e < s):
        t = t[:s]
        if not notice_inserted:
            t += " <span class='badge-thinking'>생각 중…</span> "
            notice_inserted = True

    # 5) If text begins with "think: ..." lines before a blank line, strip them
    m = re.match(r"(?is)^\s*(?:think|analysis|thoughts)\s*[:>].*?(?:\n\s*\n|$)", t)
    if m:
        t = t[m.end():]
        if not notice_inserted:
            t = " <span class='badge-thinking'>생각 중…</span> " + t
            notice_inserted = True

    return t, notice_inserted

def run_agent_stream(user_text: str, ctx: Dict[str, Any], debug: bool = False):
    agent = get_agent()
    full_prompt = f"{user_text}\n\n{context_as_text(ctx)}"
    st.session_state.last_debug = {"events": [], "error": None, "timing": {}}
    t0 = time.perf_counter()
    first_token_time = None
    try: agent.debug_mode = bool(debug)
    except Exception: pass
    try:
        for ev in agent.run(full_prompt, stream=True):
            content = getattr(ev, "content", None)
            event_name = getattr(ev, "event", "")
            if content and (event_name == "RunResponseContent" or isinstance(content, str)):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    st.session_state.last_debug["timing"]["ttft_sec"] = round(first_token_time - t0, 3)
                yield content
        t1 = time.perf_counter()
        st.session_state.last_debug["timing"]["total_sec"] = round(t1 - t0, 3)
        if first_token_time is not None:
            st.session_state.last_debug["timing"]["stream_sec"] = round(t1 - first_token_time, 3)
    except Exception as e:
        st.session_state.last_debug["error"] = {"message": str(e), "traceback": traceback.format_exc()}
        yield f"\n\n[에러] {e}"


# ==================== Session Init ====================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_debug", {"events": [], "error": None, "timing": {}})
st.session_state.setdefault("sim_params", DEFAULT_PARAM_SCHEMA.copy())
st.session_state.setdefault("context", {"customer": None, "accounts": [], "dc_contracts": []})
st.session_state.setdefault("selected_customer", None)
st.session_state.setdefault("agent_cfg", {
    "provider":"ollama","model_id":"qwen3-h2o2-14b",
    "options":{"temperature":0.3,"top_p":0.9,"top_k":40,"repeat_penalty":1.05,"num_ctx":16384,"num_predict":-1,"num_batch":512,"num_thread":0},
    "keep_alive":"2h","search_knowledge": True
})


# ==================== Layout ====================
left, midsep, right = st.columns([0.48, 0.02, 0.50])

# -------- LEFT --------
with left:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("고객/계좌 정보")

    # 고객 선택 (selectbox)
    if "df_cust" not in st.session_state:
        st.session_state.df_cust = load_customers_from_db()
    df_cust = st.session_state.df_cust
    all_names = df_cust["고객 이름"].dropna().astype(str).tolist()
    name_selected = st.selectbox("고객 이름을 선택하세요. 검색도 가능합니다",
                                 options=[""] + sorted(all_names), index=0, key="customer_select")

    # 고객/계좌 로딩
    if name_selected:
        filtered_cust = df_cust[df_cust["고객 이름"] == name_selected]
    else:
        filtered_cust = df_cust.iloc[0:0]

    if not filtered_cust.empty:
        r = filtered_cust.iloc[0]
        newly_selected_cust = r["_customer_id"]

        if newly_selected_cust != st.session_state.get("selected_customer"):
            st.session_state.selected_customer = newly_selected_cust

            # 1) 해당 고객의 모든 계좌 로드
            st.session_state.df_acct = load_accounts_from_db(newly_selected_cust)

            # 2) DC 계약 재조회
            df_acct_all = st.session_state.df_acct
            dc_acct_ids = df_acct_all.loc[
                (df_acct_all["계좌 유형"] == "DC") & (df_acct_all["_account_id"].notna()),
                "_account_id"
            ].astype(str).tolist()
            st.session_state.df_dc = load_dc_contracts_from_db(dc_acct_ids)

            # 3) 컨텍스트 재생성
            st.session_state.context = build_context_from_selection()
            st.rerun()

        # 고객 요약 카드
        st.markdown(f"""
        <div class="summary-card">
          <table>
            <tr><td style="color:#6c757d;width:110px;">고객 번호</td><td><b>{r["고객 번호"]}</b></td></tr>
            <tr><td style="color:#6c757d;">고객 이름</td><td>{r["고객 이름"]}</td></tr>
            <tr><td style="color:#6c757d;">생년월일</td><td>{r["생년월일"]}</td></tr>
            <tr><td style="color:#6c757d;">연령대</td><td>{r["연령대"]}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    else:
        if st.session_state.get("selected_customer") is not None:
            st.session_state.selected_customer = None
            st.session_state.df_acct = pd.DataFrame()
            st.session_state.df_dc = pd.DataFrame()
            st.session_state.context = {"customer": None, "accounts": [], "dc_contracts": []}
            st.rerun()
        st.info("고객을 선택하세요.")

    st.markdown("---")

    # ② 계좌 정보
    st.caption("② 계좌 정보")
    df_acct = st.session_state.get("df_acct", pd.DataFrame()).copy()
    if not df_acct.empty:
        col_chart, col_grid = st.columns([1, 1], gap="large")

        with col_chart:
            st.markdown("**계좌 유형별 평가금액 분포**")
            tmp = df_acct.copy()
            tmp["평가적립금"] = pd.to_numeric(tmp["평가적립금"], errors="coerce").fillna(0)
            grp = tmp.groupby("계좌 유형", dropna=False)["평가적립금"].sum().reset_index().sort_values("평가적립금", ascending=True)
            fig = px.bar(grp, x="평가적립금", y="계좌 유형", orientation="h", color="계좌 유형", text="평가적립금")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="계좌 유형")
            st.plotly_chart(fig, use_container_width=True)

        with col_grid:
            view_cols = ["계좌 번호","계좌 유형","상품코드","개설일자","평가적립금"]
            aggrid_table(
                df_acct[view_cols].copy(), key=GRID_KEYS["acct"], selection_mode="none", height=320,
                enable_filter=True, fit_columns_on_load=False, allow_horizontal_scroll=True
            )

        st.markdown('<div class="small-note">해당 고객의 모든 계좌가 컨텍스트에 포함됩니다. DC 계약은 DC 유형 계좌에 한해 연결됩니다.</div>', unsafe_allow_html=True)
    else:
        st.info("고객 선택 시 계좌 정보가 표시됩니다.")

    # ③ DC 계약
    st.markdown("---")
    st.caption("③ DC 계약 정보")
    df_dc = st.session_state.get("df_dc", pd.DataFrame())
    if df_dc is not None and not df_dc.empty:
        view_cols = ["계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액"]
        use_cols = [c for c in view_cols if c in df_dc.columns]
        aggrid_table(
            df_dc[use_cols].copy(), key=GRID_KEYS["dc"], selection_mode="none", height=260,
            enable_filter=True, fit_columns_on_load=False, allow_horizontal_scroll=True
        )
    else:
        st.info("선택된 고객의 DC 유형 계좌에 매핑되는 DC 계약 데이터가 없습니다.")

    st.markdown('</div>', unsafe_allow_html=True)


# -------- MID SEP --------
with midsep:
    st.markdown('<div class="v-sep"></div>', unsafe_allow_html=True)


# -------- RIGHT --------
with right:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("챗봇 · 시뮬레이션")

    # Debug 모드 토글
    debug_on = st.toggle("디버그 모드", value=False)

    # 1) (DEBUG 전용) LLM 옵션/진단 패널 — Context Builder 그대로 존재, 4컬럼 압축 배치
    if debug_on:
        with st.expander("⚙️ LLM 성능/진단 (Debug 전용)", expanded=False):
            cfg = st.session_state.get("agent_cfg") or {}
            prov = st.selectbox(
                "Provider", options=["ollama", "openai"],
                index=0 if cfg.get("provider", "ollama") == "ollama" else 1
            )

            # 공통 기본값 헬퍼
            def _opt(name, default):
                return (cfg.get("options", {}) or {}).get(name, default)

            if prov == "openai":
                # OpenAI 기본값 보정 (num_predict < 1 방지)
                raw_np = _opt("num_predict", 1024)
                try:
                    default_np = int(raw_np)
                except Exception:
                    default_np = 1024
                if default_np < 1:
                    default_np = 1024

                c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
                with c1:
                    model_id = st.selectbox("OpenAI 모델", options=["gpt-4o-mini"], index=0)
                    temperature = st.number_input("temperature", 0.0, 2.0, float(_opt("temperature", 0.3)), 0.05)
                with c2:
                    top_p = st.number_input("top_p", 0.1, 1.0, float(_opt("top_p", 0.9)), 0.05)
                    num_predict = st.number_input("num_predict (≥1)", min_value=1, max_value=40960, value=default_np)
                with c3:
                    num_thread = st.number_input("num_thread (0=auto)", min_value=0, max_value=32, value=int(_opt("num_thread", 0)))
                    rag_on = st.checkbox("지식 검색 사용 (RAG)", value=bool(cfg.get("search_knowledge", True)))
                with c4:
                    st.caption("OpenAI는 num_ctx/top_k/repeat_penalty/num_batch 미지원")

                if st.button("설정 적용", use_container_width=True):
                    st.session_state.agent_cfg = {
                        "provider": "openai",
                        "model_id": model_id,
                        "search_knowledge": bool(rag_on),
                        "keep_alive": "2h",
                        "options": {
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                            "num_predict": int(num_predict),     # ≥ 1
                            "num_thread": int(num_thread),
                        },
                    }
                    st.session_state.pop("_agent_sig", None)
                    st.success("LLM 설정을 적용했습니다.")

            else:
                # OLLAMA (qwen3-h2o2-14b 기본) — 4컬럼 x 2행 구성
                r1c1, r1c2, r1c3, r1c4 = st.columns([1.5, 1, 1, 1])
                with r1c1:
                    model_id = st.text_input("Ollama 모델 ID", value=cfg.get("model_id", "qwen3-h2o2-14b"),
                                            help="예: qwen3-h2o2-14b")
                with r1c2:
                    num_ctx = st.number_input("num_ctx", min_value=1024, max_value=32768, value=int(_opt("num_ctx", 16384)), step=1024)
                with r1c3:
                    num_predict = st.number_input("num_predict (-1=무제한)", min_value=-1, max_value=40960, value=int(_opt("num_predict", -1)))
                with r1c4:
                    num_batch = st.number_input("num_batch", min_value=1, max_value=8192, value=int(_opt("num_batch", 512)))

                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                with r2c1:
                    temperature = st.number_input("temperature", 0.0, 2.0, float(_opt("temperature", 0.3)), 0.05)
                with r2c2:
                    top_p = st.number_input("top_p", 0.1, 1.0, float(_opt("top_p", 0.9)), 0.05)
                with r2c3:
                    top_k = st.number_input("top_k", min_value=1, max_value=200, value=int(_opt("top_k", 40)))
                with r2c4:
                    repeat_penalty = st.number_input("repeat_penalty", min_value=0.8, max_value=2.0,
                                                    value=float(_opt("repeat_penalty", 1.05)), step=0.01)

                r3c1, r3c2, _, _ = st.columns(4)
                with r3c1:
                    num_thread = st.number_input("num_thread (0=auto)", min_value=0, max_value=32, value=int(_opt("num_thread", 0)))
                with r3c2:
                    rag_on = st.checkbox("지식 검색 사용 (RAG)", value=bool(cfg.get("search_knowledge", True)))

                if st.button("설정 적용", use_container_width=True):
                    st.session_state.agent_cfg = {
                        "provider": "ollama",
                        "model_id": model_id,
                        "search_knowledge": bool(rag_on),
                        "keep_alive": "2h",
                        "options": {
                            "num_ctx": int(num_ctx),
                            "num_predict": int(num_predict),      # -1 허용
                            "num_batch": int(num_batch),
                            "num_thread": int(num_thread),
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                            "top_k": int(top_k),
                            "repeat_penalty": float(repeat_penalty),
                        },
                    }
                    st.session_state.pop("_agent_sig", None)
                    st.success("LLM 설정을 적용했습니다.")

            # (선택) 간단 진단 실행 — 레이아웃 영향 최소화 (자동 스크롤 스크립트에 맡김)
            if st.button("간단 진단 실행", use_container_width=True):
                ctx_probe = {"note": "latency probe"}
                prompt_probe = "한 줄로 대답: 안녕하세요라고만 출력."
                diag_container = st.container()
                streamed = ""
                for chunk in run_agent_stream(prompt_probe, ctx_probe, debug=True):
                    streamed += chunk
                    visible, _ = mask_thoughts(streamed, notice_inserted=False)
                    with diag_container:
                        st.markdown(visible, unsafe_allow_html=True)
                t = st.session_state.get("last_debug", {}).get("timing", {})
                st.info(f"TTFT: {t.get('ttft_sec','?')}s / Stream: {t.get('stream_sec','?')}s / Total: {t.get('total_sec','?')}s")

    # 2) === 컨텍스트/파라미터 빌더 === (항상 표시, expanded=True 유지)
    with st.expander("연금 시뮬레이션 파라미터 빌더 (컨텍스트 빌더)", expanded=True):
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### 동작")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear", use_container_width=True, help="컨텍스트를 메모만 남기고 비웁니다(왼쪽 선택은 유지)."):
                    st.session_state.sim_params = DEFAULT_PARAM_SCHEMA.copy()
                    st.session_state.context = {}
                    st.success("컨텍스트를 비웠습니다(메모만 유지).")
            with c2:
                if st.button("Reset", use_container_width=True, help="왼쪽 현재 선택(고객) 기준으로 컨텍스트 복원"):
                    if st.session_state.get("selected_customer"):
                        st.session_state.df_acct = load_accounts_from_db(st.session_state.selected_customer)
                        df_acct_all = st.session_state.df_acct
                        dc_acct_ids = df_acct_all.loc[
                            (df_acct_all["계좌 유형"] == "DC") & (df_acct_all["_account_id"].notna()),
                            "_account_id"
                        ].astype(str).tolist()
                        st.session_state.df_dc = load_dc_contracts_from_db(dc_acct_ids)
                        st.session_state.context = build_context_from_selection()
                        st.success("컨텍스트를 복원했습니다.")
                    else:
                        st.session_state.context = {}
                        st.info("선택된 고객이 없습니다.")

            st.markdown("#### 수동 조정")
            p = st.session_state.setdefault("sim_params", DEFAULT_PARAM_SCHEMA.copy())
            p["notes"] = st.text_area("메모(선택)", value=p.get("notes") or "", height=100)

            # JSON 복사 버튼
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
            # JSON 미리보기: 기본 "접힘" 상태
            with st.expander("JSON 미리보기 (접힘 기본)", expanded=False):
                edit_mode = st.toggle("편집 모드(고급): 내부 스키마 JSON 직접 수정", value=False)
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
                                if not isinstance(params, dict) or (context is not None and not isinstance(context, dict)):
                                    raise ValueError("params/context는 object여야 합니다.")
                                st.session_state.sim_params = {"notes": params.get("notes", "")}
                                st.session_state.context = context if isinstance(context, dict) else {}
                                st.success("JSON을 적용했습니다.")
                            except Exception as e:
                                st.error(f"적용 실패: {e}")
                    with apply_col2:
                        st.caption("주의: 내부 키(영문) 스키마로만 편집 가능합니다.")

    # -------------------- 채팅 UI --------------------
    st.divider()
    st.markdown("#### 채팅")

    # 1) 채팅 메시지 영역 (입력창보다 위)
    chat_holder = st.container()
    with chat_holder:
        for msg in st.session_state.messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            st.chat_message(role).markdown(msg["content"], unsafe_allow_html=True)

    # 2) ✅ 입력창 '바로 위'에 앵커(스크롤 목표 지점)
    st.markdown("<div id='right-chat-input-anchor'></div>", unsafe_allow_html=True)

    # 3) 기본 입력창은 그대로 유지 (오른쪽 하단)
    user_input = st.chat_input("질문을 입력하세요. (예: 현재 컨텍스트 기반으로 DC 관련 규정 설명)")

    # 4) 페이지 렌더 직후에도 항상 앵커로 스크롤 → 입력창이 화면에 바로 나타남
    components.v1.html("""
    <script>
    const el = document.getElementById('right-chat-input-anchor');
    if (el) el.scrollIntoView({behavior: 'auto', block: 'end'});
    </script>
    """, height=0)

    # 5) 전송 처리: 스트리밍 중에도 입력창 근처로 스크롤 유지
    if user_input:
        # (1) 유저 메시지 저장/렌더
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_holder:
            st.chat_message("user").markdown(user_input)

            # (2) 컨텍스트 확정 후, 스트리밍 출력
            ctx = build_context_for_chat()
            resp_area = st.chat_message("assistant")
            placeholder = resp_area.empty()
            streamed = ""
            displayed_once_think = False

            for chunk in run_agent_stream(user_input, ctx, debug=debug_on):
                streamed += chunk
                visible, displayed_once_think = mask_thoughts(streamed, displayed_once_think)
                placeholder.markdown(visible, unsafe_allow_html=True)

            # (3) 최종 텍스트 저장
            final_visible, _ = mask_thoughts(streamed, displayed_once_think)
            st.session_state.messages.append({"role": "assistant", "content": final_visible})


    st.markdown('</div>', unsafe_allow_html=True)  # 오른쪽 패널 끝
