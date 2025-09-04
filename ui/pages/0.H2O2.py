# st_sample.py
# -*- coding: utf-8 -*-
# 퇴직연금 RAG + 시뮬레이터 (갱신: 2025-09-04)
# - Clear/Reset 즉시 JSON 미리보기 갱신(st.rerun 사용)
# - 수동 조정은 동작 버튼 아래로 이동(세로 배열)
# - JSON 미리보기는 오른쪽 컬럼으로 이동
# - JSON 다운로드 제거, "JSON 복사"는 동작 섹션 쪽 버튼으로 표시
# - run_pension_simulator 더미 / DEFAULT_PARAM_SCHEMA는 notes만 유지

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

from sqlalchemy.sql import bindparam
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, text, event
import pgvector.sqlalchemy

try:
    from pgvector.psycopg import register_vector
except Exception:
    register_vector = None

# (선택) agno 관련 더미/예시
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

def koreanize_dict(d: Dict[str, Any], kmap: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[kmap.get(k, k)] = v
    return out

def koreanize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = payload.get("context") or {}
    cust = ctx.get("customer")
    accts = ctx.get("accounts") or []
    dc = ctx.get("dc_contract")
    return {
        "파라미터": payload.get("params"),
        "컨텍스트": {
            "고객": koreanize_dict(cust, KMAP_CUSTOMER) if cust else None,
            "계좌들": [koreanize_dict(a, KMAP_ACCOUNT) for a in accts],
            "DC 계약": koreanize_dict(dc, KMAP_DC) if dc else None,
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


# ==================== Demo/DB Loaders ====================
@st.cache_data(ttl=60)
def load_customers_from_db() -> pd.DataFrame:
    # 데모용: 로컬 생성 (실환경이면 DB 조회로 교체)
    return pd.DataFrame(
        [
            {"_customer_id": "C001", "고객 번호": "C001", "고객 이름": "홍길동", "생년월일": "1985-01-01", "연령대": "40대"},
            {"_customer_id": "C002", "고객 번호": "C002", "고객 이름": "김영희", "생년월일": "1990-06-12", "연령대": "30대"},
        ]
    )

@st.cache_data(ttl=60)
def load_accounts_from_db(customer_filter: Optional[Any] = None) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"_account_id": "A-1001", "_customer_id": "C001", "계좌 번호": "A-1001", "고객 번호": "C001", "계좌 유형": "DC", "상품코드": "P01", "개설일자": "2018-03-01", "평가적립금": 32500000},
            {"_account_id": "A-1002", "_customer_id": "C001", "계좌 번호": "A-1002", "고객 번호": "C001", "계좌 유형": "IRP","상품코드": "P02", "개설일자": "2020-10-02", "평가적립금": 8500000},
            {"_account_id": "A-2001", "_customer_id": "C002", "계좌 번호": "A-2001", "고객 번호": "C002", "계좌 유형": "DC", "상품코드": "P03", "개설일자": "2019-07-15", "평가적립금": 17300000},
        ]
    )
    if customer_filter is None:
        return df
    if isinstance(customer_filter, (list, tuple, set)):
        return df[df["_customer_id"].isin(list(customer_filter))].reset_index(drop=True)
    return df[df["_customer_id"] == customer_filter].reset_index(drop=True)

@st.cache_data(ttl=60)
def load_dc_contracts_from_db(account_filter=None) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {"_ctrt_no": "A-1001", "계약번호": "A-1001", "근무처명": "길동전자", "입사일자": "2015-02-10", "중간정산일자": None, "제도가입일자": "2015-03-01",
             "부담금납입원금": 24000000, "운용손익금액": 8500000, "평가적립금합계금액": 32500000},
            {"_ctrt_no": "A-2001", "계약번호": "A-2001", "근무처명": "한빛제약", "입사일자": "2018-05-21", "중간정산일자": None, "제도가입일자": "2018-06-01",
             "부담금납입원금": 14000000, "운용손익금액": 3300000, "평가적립금합계금액": 17300000},
        ]
    )
    if account_filter is None:
        return base
    ids = account_filter if isinstance(account_filter, (list, tuple, set)) else [account_filter]
    return base[base["_ctrt_no"].isin(list(ids))].reset_index(drop=True)


# ==================== Simple Agent (더미 스트리밍) ====================
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
    engine = create_engine("postgresql+psycopg://user:pass@localhost:5432/db")  # 데모
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
    agent = Agent(system_message=sys, model=model, tools=[run_pension_simulator],
                  markdown=True, knowledge=make_knowledge_base(), search_knowledge=True,
                  enable_agentic_knowledge_filters=True, show_tool_calls=True, debug_mode=False)
    return agent

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

# 데이터 로드(데모)
if "demo_df" not in st.session_state:
    st.session_state.demo_df = load_customers_from_db()
if "acct_df" not in st.session_state:
    st.session_state.acct_df = load_accounts_from_db()
st.session_state.dc_df = load_dc_contracts_from_db(st.session_state.get("selected_accounts") or None)


# ==================== Layout ====================
left, midsep, right = st.columns([0.46, 0.02, 0.52])

# -------- LEFT --------
with left:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("고객/계좌 정보")

    # 고객 선택
    st.caption("① 고객을 하나 선택하세요 (싱글 선택)")
    cust_df = st.session_state.demo_df
    st.dataframe(cust_df[["고객 번호", "고객 이름", "생년월일", "연령대"]], use_container_width=True, hide_index=True, height=180)
    st.session_state.selected_customer = st.selectbox(
        "고객 선택", options=[""] + cust_df["_customer_id"].tolist(),
        index=1 if cust_df.shape[0] else 0, help="고객을 선택하면 우측 Reset 시 컨텍스트가 갱신됩니다."
    )

    # 계좌 표/선택
    st.divider()
    st.caption("② 계좌 선택")
    only_selected = st.checkbox("선택 고객의 계좌만 보기", value=True)
    if only_selected:
        if st.session_state.selected_customer:
            current_acct_df = load_accounts_from_db(st.session_state.selected_customer)
        else:
            current_acct_df = pd.DataFrame(columns=st.session_state.acct_df.columns)  # 빈
    else:
        current_acct_df = load_accounts_from_db()
    st.session_state.acct_df = current_acct_df

    st.dataframe(
        current_acct_df[["계좌 번호","고객 번호","계좌 유형","상품코드","개설일자","평가적립금"]],
        use_container_width=True, hide_index=True, height=220
    )
    st.session_state.selected_accounts = st.multiselect(
        "계좌 선택(멀티)", options=current_acct_df["_account_id"].tolist(), default=current_acct_df["_account_id"].tolist()[:1]
    )

    # 파이 차트
    st.subheader("계좌 유형별 평가적립금")
    pie_df = current_acct_df.copy()
    if pie_df.empty or pie_df["평가적립금"].fillna(0).sum() == 0:
        st.info("표시할 평가적립금이 없습니다.")
    else:
        grp = pie_df.groupby("계좌 유형", dropna=False)["평가적립금"].sum().reset_index().sort_values("평가적립금", ascending=False)
        fig = px.pie(grp, names="계좌 유형", values="평가적립금", hole=0.4)
        fig.update_traces(textinfo="percent+label", textposition="inside", hovertemplate="%{label}<br>%{value:,}원<br>%{percent}")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="계좌 유형")
        st.plotly_chart(fig, use_container_width=True)

    # DC 계약
    st.divider()
    st.caption("③ DC 계약 (계약번호=계좌번호 연결)")
    acct_ids = st.session_state.get("selected_accounts", [])
    st.session_state.dc_df = load_dc_contracts_from_db(acct_ids if acct_ids else None)
    dc_df = st.session_state.dc_df
    if dc_df is not None and not dc_df.empty:
        st.dataframe(dc_df[["계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액"]],
                     use_container_width=True, hide_index=True, height=200)
    else:
        st.info("표시할 DC 계약 데이터가 없습니다.")

    st.markdown('</div>', unsafe_allow_html=True)


# -------- MID SEP --------
with midsep:
    st.markdown('<div class="v-sep"></div>', unsafe_allow_html=True)


# -------- RIGHT --------
with right:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("챗봇 · 시뮬레이션")

    debug_on = st.toggle("디버그 모드", value=False)

    # ---- 파라미터 빌더 ----
    with st.expander("연금 시뮬레이션 파라미터 빌더", expanded=True):
        # 1) 좌: 동작(버튼/복사/수동조정), 우: JSON 미리보기
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### 동작")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear", use_container_width=True, help="컨텍스트를 완전히 비웁니다."):
                    st.session_state.context = {"customer": None, "accounts": [], "dc_contract": None}
                    st.rerun()  # ▶ 즉시 미리보기 갱신
            with c2:
                if st.button("Reset", use_container_width=True, help="왼쪽 선택 기준으로 컨텍스트를 다시 세팅합니다."):
                    # 좌측 선택을 이용해 컨텍스트 재구성
                    # (선택 고객/계좌/계약을 현재 상태에서 읽어와 생성)
                    selected_customer = st.session_state.get("selected_customer")
                    selected_accounts = st.session_state.get("selected_accounts", [])
                    # 고객
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
                    # 계좌
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
                                "evlu_acca_smtl_amt": int(r["평가적립금"]) if pd.notna(r["평가적립금"]) else 0,
                            })
                    # DC 계약: 첫 번째 DC 계좌 기준
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
                                    "almt_pymt_prca": int(r["부담금납입원금"]) if pd.notna(r["부담금납입원금"]) else 0,
                                    "utlz_pfls_amt": int(r["운용손익금액"]) if pd.notna(r["운용손익금액"]) else 0,
                                    "evlu_acca_smtl_amt": int(r["평가적립금합계금액"]) if pd.notna(r["평가적립금합계금액"]) else 0,
                                }
                    st.session_state.context = {"customer": cust, "accounts": accts, "dc_contract": dc}
                    st.rerun()  # ▶ 즉시 미리보기 갱신

            st.markdown("#### 수동 조정")
            p = st.session_state.sim_params
            # 세로 배열, notes만 유지
            p["notes"] = st.text_area("메모(선택)", value=p.get("notes") or "", height=100)

        # 2) (입력 반영 후) 공통 payload/표시 생성
        payload_preview = {"params": st.session_state.sim_params, "context": st.session_state.context}
        # 오른쪽: JSON 미리보기 (한글 라벨 보기 토글)
        with col_right:
            st.markdown("#### JSON 미리보기")
            show_korean = st.checkbox("표시용(한글 라벨)로 보기", value=True, key="show_kor_preview")
            display_payload = koreanize_payload(payload_preview) if show_korean else payload_preview
            st.json(display_payload)

        # 3) 하단 행: 좌측에 "JSON 복사" 버튼(동작 섹션 쪽), 우측은 비움
        json_str = to_json_str(display_payload)
        _json_for_js = json_str.replace("\\", "\\\\").replace("`", "\\`")

        c_left, c_right = st.columns([1, 1])
        with c_left:
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
        with c_right:
            st.write("")

    st.divider()

    # ---------- 채팅 ----------
    # 히스토리 렌더
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        st.chat_message(role).markdown(msg["content"])

    # 큐 처리
    queued = st.session_state.pop("queued_user_input", None)
    queued_ctx = st.session_state.pop("queued_context", None)
    if queued:
        st.session_state.messages.append({"role": "user", "content": queued})
        st.chat_message("user").markdown(queued)
        ctx = queued_ctx or st.session_state.context
        resp_area = st.chat_message("assistant")
        placeholder = resp_area.empty()
        streamed = ""
        for chunk in (word + " " for word in f"질문: {queued}\n\n(컨텍스트 요약) 고객:{bool(ctx.get('customer'))} 계좌:{len(ctx.get('accounts', []))} DC계약:{bool(ctx.get('dc_contract'))}".split()):
            streamed += chunk
            placeholder.markdown(streamed)
        st.session_state.messages.append({"role": "assistant", "content": streamed})

    # 입력창은 항상 맨 아래
    st.markdown("---")
    user_input = st.chat_input("질문을 입력하세요. (예: 현재 컨텍스트 기반으로 DC 관련 규정 설명)")
    if user_input:
        st.session_state["queued_user_input"] = user_input
        st.session_state["queued_context"] = st.session_state.context
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
