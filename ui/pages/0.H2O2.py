# st_sample.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import traceback
import inspect
from typing import Any, Dict, Optional, List
from urllib.parse import quote_plus, urlencode

import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid, GridUpdateMode
import plotly.express as px

from sqlalchemy.sql import bindparam
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, text, event
import pgvector.sqlalchemy
from pgvector.psycopg import register_vector

# agno / RAG
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama    # 예: Ollama(id="llama3.1")
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.agent import AgentKnowledge


# ==================== Page Config & Styles ====================
st.set_page_config(page_title="H2O2 - 한투 퇴직마스터", layout="wide")

st.markdown("""
<style>
/* 페이지 전체 상단 여백 줄이기 */
.main .block-container { padding-top: 0.2rem; }

/* 중앙 타이틀 */
.center-title {
  text-align:center; margin: .3rem 0 .7rem 0; font-size: 1.8rem; font-weight: 700;
}

/* 더 얇은 패널 + 플러시 탑(상단 여백 제거) 옵션 */
.panel-soft {
  padding: 12px 14px; border-radius: 12px; background: #ffffff;
  border: 1px solid rgba(0,0,0,0.04); box-shadow: 0 1px 2px rgba(0,0,0,.03);
}
.panel-soft.flush-top { padding-top: 0; }              /* ⬅️ 상단 패딩 제거 */
.panel-soft > :first-child { margin-top: 0 !important; } /* ⬅️ 첫 요소의 위쪽 마진 제거 */

/* 얇은 세로 구분선 */
.v-sep {
  border-left: 1px solid #e9ecef;
  height: calc(100vh - 180px);
  margin: 8px 6px;
}

/* Streamlit 기본 헤더 영역(툴바) 숨김이 필요하면 주석 해제
.stApp header { display: none; }
*/
</style>
<div class="center-title">한투 퇴직마스터</div>
""", unsafe_allow_html=True)



# ==================== Constants ====================
DEFAULT_PARAM_SCHEMA = {
    "customer_id": None,
    "customer_name": None,
    "customer_age": None,
    "risk_level": None,
    "retirement_age": 65,
    "current_balance": 0,
    "monthly_contribution": 0,
    "expected_return_pct": 4.0,
    "inflation_pct": 2.0,
    "years_to_retirement": None,
    "notes": ""
}

GRID_KEYS = {
    "demo": "grid_demo_pg_v3",
    "acct": "grid_acct_pg_v3",
    "dc_ctrt": "grid_dc_contracts",
}
BUTTON_KEYS = {
    "fill_customer": "btn_fill_customer_v3",
    "sum_accounts": "btn_sum_accounts_v3",
    "estimate_monthly": "btn_estimate_monthly_v3",
    "reset_params": "btn_reset_params_v3",
    "send_json": "btn_send_json_v3",
    "run_tool": "btn_run_tool_v3",
}

_KMAP_CUSTOMER = {
    "customer_id": "고객 번호",
    "customer_name": "고객 이름",
    "birth": "생년월일",
    "age_band": "연령대",   # 있으면 적용, 없으면 무시
}
_KMAP_ACCOUNT = {
    "account_id": "계좌 번호",
    "customer_id": "고객 번호",
    "product_type": "계좌 유형",
    "prod_code": "상품코드",
    "opened_at": "개설일자",
    "evlu_acca_smtl_amt": "평가적립금",
}
_KMAP_DC = {
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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def to_json_str(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)

def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

# ==================== String Utils ====================

def _map_keys(obj, kmap):
    """dict / list[dict]의 키를 kmap으로 바꾼 복사본을 반환 (원본 불변)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {kmap.get(k, k): _map_keys(v, kmap) if isinstance(v, (dict, list)) else v for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_keys(x, kmap) if isinstance(x, (dict, list)) else x for x in obj]
    return obj  # primitive
 
def koreanize_context(ctx: dict) -> dict:
    """context 사본을 한글 라벨로 변환 (customer, accounts, dc_contract 각각 처리)."""
    if not isinstance(ctx, dict):
        return ctx
    out = dict(ctx)  # shallow copy
 
    if "customer" in out:
        out["고객"] = _map_keys(out["customer"], _KMAP_CUSTOMER)
        out.pop("customer", None)
 
    if "accounts" in out:
        # accounts는 list[dict]
        out["계좌"] = _map_keys(out["accounts"], _KMAP_ACCOUNT)
        out.pop("accounts", None)
 
    if "dc_contract" in out:
        out["DC 계약"] = _map_keys(out["dc_contract"], _KMAP_DC)
        out.pop("dc_contract", None)
 
    return out
 
def koreanize_payload(payload: dict) -> dict:
    """payload 사본에서 context만 한글 라벨로 바꾼 표시용 버전 생성."""
    if not isinstance(payload, dict):
        return payload
    disp = dict(payload)
    if "context" in disp:
        disp["context"] = koreanize_context(disp["context"])
    return disp


def _to_py_scalar(v):
    """NumPy 스칼라 → 파이썬 내장형으로 변환"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

def _to_py_list(vals):
    return [_to_py_scalar(x) for x in vals]

# 문자열 → (JSON / 숫자 / 원문 문자열) 자동 파서
def _parse_value_auto(s: str):
    s = (s or "").strip()
    if s == "":
        return ""
    # JSON 객체/배열 시도
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass
    # 숫자 시도 (int → float 순)
    try:
        if s.isdigit() or (s[0] == "-" and s[1:].isdigit()):
            return int(s)
        return float(s)
    except Exception:
        pass
    # 그 외 문자열
    return s

# (권장) psycopg3 커넥션에 pgvector 어댑터 등록
try:
    from pgvector.psycopg import register_vector
except Exception:
    register_vector = None

# 안전한 중복 방지
if not globals().get("_VEC_GLOBAL_HOOKS_INSTALLED", False):

    @event.listens_for(Engine, "connect")
    def _vec_register_on_connect(dbapi_connection, connection_record):
        # psycopg3 연결에 한해 list[float] → vector 바인딩 지원
        if register_vector is not None:
            try:
                register_vector(dbapi_connection)
            except Exception:
                # 드라이버/환경에 따라 이미 등록되었거나 미지원일 수 있음 → 무시
                pass

    @event.listens_for(Engine, "before_cursor_execute")
    def _vec_force_cast_before_exec(conn, cursor, statement, parameters, context, executemany):
        """
        PgVector가 생성한 하이브리드/벡터 쿼리에서 오른쪽 피연산자가 문자열로 바인딩되어
        '<=> unknown' 에러가 나는 경우, 실행 직전에 ::vector를 주입한다.

        치환 대상 예시:
          - "... embedding <=> %(embedding_1)s ..."  →  "... embedding <=> %(embedding_1)s::vector ..."
          - "... embedding <=> %s ..."               →  "... embedding <=> %s::vector ..."
          - "... embedding <=> $1 ..."               →  "... embedding <=> $1::vector ..."   (일부 드라이버)

        안전을 위해 스키마/테이블/컬럼명 조건을 걸어 영향 범위를 좁힌다.
        필요 시 아래 조건을 환경에 맞게 수정하세요.
        """
        # 빠른 필터: 해당 테이블/연산자 포함 & 이미 캐스팅 안되어 있을 때만
        if "pension_knowledge" in statement and "<=>" in statement and "::vector" not in statement:
            # embedding 컬럼이 명시되지 않은 경우도 있을 수 있으나, 과도 캐스팅을 피하려면 컬럼명 조건을 유지
            if "embedding <=>" in statement or ".embedding <=>" in statement:
                # named/pyformat (%(name)s), positional (%s), dollar ($1) 을 모두 처리
                statement = re.sub(r"(<=>\s*)(%\([^)]+\)s)(?!::vector)", r"\1\2::vector", statement)
                statement = re.sub(r"(<=>\s*)(%s)(?!::vector)",            r"\1\2::vector", statement)
                statement = re.sub(r"(<=>\s*)(\$\d+)(?!::vector)",         r"\1\2::vector", statement)

        return statement, parameters

    globals()["_VEC_GLOBAL_HOOKS_INSTALLED"] = True
# === /전역 훅 끝 ===

# ==================== DB Utils & Loaders ====================

def _safe_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

def _safe_schema(schema: Optional[str]) -> str:
    if not schema:
        return "public"
    schema = schema.strip()
    return schema if re.fullmatch(r"[A-Za-z0-9_]+", schema) else "public"

def _get_pg_conn_str() -> str:
    host = _safe_secret("PG_HOST", "localhost") or "localhost"
    port = _safe_secret("PG_PORT", "5432") or "5432"
    db   = _safe_secret("PG_DB", "postgres") or "postgres"
    user = _safe_secret("PG_USER", "postgres") or "postgres"
    pwd  = _safe_secret("PG_PASSWORD", None)
    if not pwd:
        raise RuntimeError("PG_PASSWORD가 설정되지 않았습니다. 환경변수 또는 .streamlit/secrets.toml에 PG_PASSWORD를 설정하세요.")
    schema = _safe_schema(_safe_secret("PG_SCHEMA", "public"))
    user_q = quote_plus(user); pwd_q = quote_plus(pwd)
    host_q = quote_plus(host); db_q = quote_plus(db)
    query = urlencode({"options": f"-csearch_path={schema}"})
    return f"postgresql+psycopg2://{user_q}:{pwd_q}@{host_q}:{port}/{db_q}?{query}".replace("psycopg2", "psycopg")

def _make_engine_with_schema():
    return create_engine(_get_pg_conn_str(), pool_pre_ping=True)

@st.cache_data(ttl=60)
def load_customers_from_db() -> pd.DataFrame:
    engine = _make_engine_with_schema()
    sql = text("""SELECT customer_id, customer_name, brth_dt, age_band FROM kis_customers ORDER BY customer_id""")
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn)
    if "brth_dt" in df.columns:
        df["brth_dt"] = pd.to_datetime(df["brth_dt"], errors="coerce").dt.date
    df.rename(columns={
        "customer_id":"고객 번호","customer_name":"고객 이름","brth_dt":"생년월일","age_band":"연령대"
    }, inplace=True)
    df["_customer_id"] = df["고객 번호"]
    return df

@st.cache_data(ttl=60)
def load_accounts_from_db(customer_filter: Optional[Any] = None) -> pd.DataFrame:
    """customer_filter: None | scalar | list/tuple/set (IN 절)"""
    engine = _make_engine_with_schema()
    base_sql = """SELECT account_id, customer_id, acnt_type, prd_type_cd, acnt_bgn_dt, acnt_evlu_amt FROM kis_accounts"""
    params: Dict[str, Any] = {}
    if customer_filter is None:
        stmt = text(base_sql + " ORDER BY account_id")
    else:
        if isinstance(customer_filter, (list, tuple, set)):
            #cids = tuple(sorted(_to_py_list(list(customer_filter))))  # ✅ 정렬된 튜플로 캐시 키 안정화
            stmt = text(base_sql + " WHERE customer_id IN :cids ORDER BY account_id").bindparams(
                bindparam("cids", expanding=True)
            )
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
    kis_dc_contract 로드.
    account_filter: None | scalar | list/tuple/set (ctrt_no IN (...))
    ctrt_no(계약번호) = 계좌번호(account_id)와 연결됨.
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
            if len(ids) == 0:
                # 빈 컬렉션 → 빈 DF
                cols = ["ctrt_no","odtp_name","etco_dt","midl_excc_dt","sst_join_dt",
                        "almt_pymt_prca","utlz_pfls_amt","evlu_acca_smtl_amt"]
                df = pd.DataFrame(columns=cols)
                return _rename_dc_columns(df)
            stmt = text(base_sql + " WHERE ctrt_no IN :ids ORDER BY ctrt_no").bindparams(
                bindparam("ids", expanding=True)
            )
            params["ids"] = ids
        else:
            cid = _to_py_scalar(account_filter)
            stmt = text(base_sql + " WHERE ctrt_no = :id ORDER BY ctrt_no")
            params["id"] = cid

    with engine.begin() as conn:
        df = pd.read_sql(stmt, conn, params=params)
    return _rename_dc_columns(df)

def _rename_dc_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 한글 컬럼명 적용 + 내부 키 보조 컬럼
    df = df.copy()
    # 날짜/숫자 정리
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
    # 내부 키
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
        try:
            gob.configure_side_bar()
        except Exception:
            pass

    grid_options = gob.build()

    # ✅ 선택/필터/모델 변경 모두 즉시 이벤트 발생
    update_mode = (
        GridUpdateMode.SELECTION_CHANGED
        | GridUpdateMode.FILTERING_CHANGED
        | GridUpdateMode.MODEL_CHANGED
    )

    return AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=update_mode,         # ← 여기만 바꿔도 체감이 큽니다
        height=height,
        key=key,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
    )

def get_first_value_from_selection(selection, key: str):
    if selection is None:
        return None
    if isinstance(selection, list):
        if len(selection) == 0: return None
        first = selection[0]
        return first.get(key) if isinstance(first, dict) else None
    if isinstance(selection, pd.DataFrame):
        if selection.empty or key not in selection.columns: return None
        return selection.iloc[0][key]
    return None

def get_all_values_from_selection(selection, key: str):
    if selection is None:
        return []
    if isinstance(selection, list):
        vals = [row.get(key) for row in selection if isinstance(row, dict) and key in row]
        return [_to_py_scalar(v) for v in vals]
    if isinstance(selection, pd.DataFrame):
        if key in selection.columns:
            return [_to_py_scalar(v) for v in selection[key].dropna().tolist()]
    return []

# 컨텍스트 공통 빌더
def build_context_for_chat():
    df_acct = st.session_state.acct_df
    sel_ids = [_to_py_scalar(x) for x in st.session_state.selected_accounts]
    if sel_ids:
        acct_ctx_df = df_acct[df_acct["_account_id"].isin(sel_ids)]
    else:
        acct_ctx_df = df_acct
    return {
        "customer": (
            st.session_state.demo_df[
                st.session_state.demo_df["_customer_id"] == st.session_state.selected_customer
            ][["고객 번호", "고객 이름", "생년월일", "연령대"]]
            .to_dict(orient="records")[0]
            if st.session_state.selected_customer else None
        ),
        "accounts": acct_ctx_df[
            ["계좌 번호", "고객 번호", "계좌 유형", "상품코드", "개설일자", "평가적립금"]
        ].to_dict(orient="records"),
    }

# ==================== Pension Tools ====================
@tool
def run_pension_simulator(params: dict) -> dict:
    import httpx
    url = _safe_secret("PENSION_SIM_URL", "")
    api_key = _safe_secret("PENSION_SIM_API_KEY", "")
    if url:
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return {"source": "api", "ok": True, "result": data}
        except Exception:
            pass
    # fallback: 내부 계산
    def _fv_with_inflation(current_balance, monthly_contribution, years, r_pct, inflation_pct):
        n_months = max(0, int(years) * 12)
        r = float(r_pct) / 100.0
        i = float(inflation_pct) / 100.0
        if n_months == 0:
            return {"future_value_nominal": current_balance, "future_value_real": current_balance, "assumptions": {"months": 0}}
        rm = r / 12.0
        im = i / 12.0
        fv_lump = current_balance * ((1 + rm) ** n_months)
        fv_ann = (monthly_contribution * (((1 + rm) ** n_months - 1) / rm)) if rm != 0 else (monthly_contribution * n_months)
        fv_nominal = fv_lump + fv_ann
        fv_real = fv_nominal / ((1 + im) ** n_months)
        return {"future_value_nominal": round(fv_nominal), "future_value_real": round(fv_real), "assumptions": {"months": n_months}}
    years = params.get("years_to_retirement")
    if years is None and params.get("customer_age") is not None and params.get("retirement_age") is not None:
        years = max(0, int(params["retirement_age"]) - int(params["customer_age"]))
    res = _fv_with_inflation(
        float(params.get("current_balance", 0) or 0),
        float(params.get("monthly_contribution", 0) or 0),
        int(years or 0),
        float(params.get("expected_return_pct", 4.0) or 0.0),
        float(params.get("inflation_pct", 2.0) or 0.0),
    )
    return {"source": "local", "ok": True, "result": res, "echo_params": params}

@tool
def pretty_print_simulation(result: dict) -> str:
    try:
        res = result.get("result", result)
        fv_nom = int(res.get("future_value_nominal", 0))
        fv_real = int(res.get("future_value_real", 0))
        months = res.get("assumptions", {}).get("months")
        return f"**시뮬레이션 요약**\n- 적립 기간: 약 {months}개월\n- 미래가치(명목): {fv_nom:,}원\n- 미래가치(실질): {fv_real:,}원\n"
    except Exception:
        return "시뮬레이션 결과 요약 중 문제가 발생했습니다."


# ==================== RAG: VectorOnlyKnowledge ====================
# 하이브리드/벡터/텍스트 검색을 직접 수행하는 대체 백엔드

class VectorOnlyKnowledge(AgentKnowledge):
    """pgvector 테이블만 검색용으로 사용하는 AgentKnowledge 구현"""
    def __init__(self, vector_db, filters=None, name: str = "pension_knowledge"):
        super().__init__(vector_db=vector_db, filters=filters, name=name)
    @property
    def document_lists(self):
        return []  # 원본 문서 소스 없음
    def load(self):
        return self

def make_knowledge_base() -> AgentKnowledge:
    table = os.getenv("AGNO_KG_TABLE", "pension_knowledge")
    search = os.getenv("AGNO_KG_SEARCH", "hybrid").lower()
    search_type = SearchType.hybrid if search == "hybrid" else (SearchType.fulltext if search == "fulltext" else SearchType.vector)

    # 1) psycopg3 엔진 생성
    engine = _make_engine_with_schema()

    # 2) PgVector는 그대로 사용 (db_url 대신 engine 전달)
    embedder = OllamaEmbedder(id="openhermes")
    vector_db = PgVector(
        db_engine=engine,                 # ★ 중요: engine을 넘겨 위 훅들이 적용되도록
        #schema="ai",
        table_name=table,
        embedder=embedder,
        search_type=search_type,
    )

    return VectorOnlyKnowledge(vector_db=vector_db, name=table)

# search() 인자 호환 (limit/k/top_k)
def _vdb_search_compat(vdb, query: str, k: int = 1, search_type=None):
    try:
        sig = inspect.signature(vdb.search)
    except Exception:
        sig = None
    tried = []
    try:
        if sig is None or "limit" in sig.parameters:
            kwargs = {"limit": k}
            if search_type is not None and (sig is None or "search_type" in sig.parameters):
                kwargs["search_type"] = search_type
            return vdb.search(query, **kwargs)
    except TypeError as e:
        tried.append(("limit", str(e)))
    try:
        if sig is None or "k" in sig.parameters:
            kwargs = {"k": k}
            if search_type is not None and (sig is None or "search_type" in sig.parameters):
                kwargs["search_type"] = search_type
            return vdb.search(query, **kwargs)
    except TypeError as e:
        tried.append(("k", str(e)))
    try:
        if sig is None or "top_k" in sig.parameters:
            kwargs = {"top_k": k}
            if search_type is not None and (sig is None or "search_type" in sig.parameters):
                kwargs["search_type"] = search_type
            return vdb.search(query, **kwargs)
    except TypeError as e:
        tried.append(("top_k", str(e)))
    raise TypeError(f"PgVector.search 인자 호환 실패: tried={tried}")

def kb_healthcheck():
    try:
        kb = make_knowledge_base()
        vdb = kb.vector_db
        res = _vdb_search_compat(vdb, "healthcheck", k=1)
        n = len(res) if isinstance(res, (list, tuple)) else (res.get("count", 1) if isinstance(res, dict) else 1)
        return True, f"pgvector search OK (hits≈{n})"
    except Exception as e:
        return False, f"pgvector search failed: {e}"

def make_agent() -> Agent:
    sys = (
        "당신은 퇴직연금 상담 어시스턴트입니다. "
        "웹 검색 대신 DB의 pension_knowledge(pgvector)와 좌측 컨텍스트만 사용해 답하세요. "
        "필요 시 툴(run_pension_simulator, pretty_print_simulation)로 계산하세요."
    )
    model = OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    model = Ollama(id="qwen3-h2o2-30b", request_params={"think": False, "keep_alive": "2h"})
    agent = Agent(
        system_message=sys,
        model=model,
        tools=[run_pension_simulator, pretty_print_simulation],
        markdown=True,
        knowledge=make_knowledge_base(),
        search_knowledge=True,
        enable_agentic_knowledge_filters=True,
        show_tool_calls=True,
        debug_mode=False,
    )
    return agent

AGENT = make_agent()

# 스트리밍 + 디버그 로깅
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
            if debug:
                try:
                    st.session_state.last_debug["events"].append({
                        "t": round(time.perf_counter() - t0, 3),
                        "event": getattr(ev, "event", None),
                        "type": type(ev).__name__,
                        "content_preview": (getattr(ev, "content", None) or "")[:200],
                        "tool": getattr(ev, "tool", None) or getattr(ev, "tool_name", None),
                        "metadata": getattr(ev, "metadata", None),
                    })
                    if len(st.session_state.last_debug["events"]) > 400:
                        st.session_state.last_debug["events"] = st.session_state.last_debug["events"][-400:]
                except Exception:
                    pass
            content = getattr(ev, "content", None)
            event_name = getattr(ev, "event", "")
            if content and (event_name == "RunResponseContent" or isinstance(content, str)):
                yield content
        st.session_state.last_debug["timing"]["total_sec"] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        st.session_state.last_debug["error"] = {"message": str(e), "traceback": traceback.format_exc()}
        yield f"\n\n[에러] {e}"

# 초기 세션 상태
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_debug", {"events": [], "error": None, "timing": {}})
st.session_state.setdefault("sim_params", DEFAULT_PARAM_SCHEMA.copy())

# ==================== Session Data Load ====================
if "demo_df" not in st.session_state:
    st.session_state.demo_df = load_customers_from_db()
if "acct_df" not in st.session_state:
    st.session_state.acct_df = load_accounts_from_db()

st.session_state.setdefault("selected_customer", None)
st.session_state.setdefault("selected_accounts", [])
st.session_state.setdefault("dc_df", pd.DataFrame())  # DC 계약 캐시


# ==================== Layout Columns ====================
left, midsep, right = st.columns([0.46, 0.02, 0.52])

with left:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)
    st.subheader("고객/계좌 정보")
    # -------- 고객 그리드: 먼저 렌더 --------
    st.caption("고객을 하나 선택하세요 (싱글 선택)")
    grid = aggrid_table(
        st.session_state.demo_df[["고객 번호", "고객 이름", "생년월일", "연령대", "_customer_id"]],
        key=GRID_KEYS["demo"],
        selection_mode="single",
        height=260,
        enable_filter=True,
        show_side_bar=False,
    )
    sel = grid.get("selected_rows", None)
    st.session_state.selected_customer = get_first_value_from_selection(sel, "_customer_id")

    # 현재 그리드에 '보이는' 고객들(필터 반영)
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

    # ✅ 계좌 DF 재조회: 선택 고객 > (선택 없음) 필터 결과 고객들 > 전체
    if only_selected:
        if st.session_state.selected_customer is not None:
            current_acct_df = load_accounts_from_db(st.session_state.selected_customer)
        elif visible_ids:
            current_acct_df = load_accounts_from_db(visible_ids)
        else:
            current_acct_df = load_accounts_from_db()
    else:
        current_acct_df = load_accounts_from_db()

    # 세션 반영(하위 그리드 및 다른 섹션에서 사용)
    st.session_state.acct_df = current_acct_df

    # -------- (이제) 차트: 갱신된 current_acct_df로 즉시 그림 --------
    st.subheader("계좌 유형별 평가적립금")
    pie_df = current_acct_df.copy()
    if pie_df.empty or pie_df["평가적립금"].fillna(0).sum() == 0:
        st.info("표시할 평가적립금이 없습니다. 고객을 선택하거나 계좌를 선택해 주세요.")
    else:
        grp = (
            pie_df.groupby("계좌 유형", dropna=False)["평가적립금"]
            .sum()
            .reset_index()
            .sort_values("평가적립금", ascending=False)
        )
        fig = px.pie(grp, names="계좌 유형", values="평가적립금", hole=0.4)
        fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            insidetextorientation="auto",
            hovertemplate="%{label}<br>%{value:,}원<br>%{percent}",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="계좌 유형")
        st.plotly_chart(fig, use_container_width=True)

    # -------- 계좌 그리드 --------
    st.caption("계좌를 선택하세요 (멀티 선택 가능)")
    grid_acct = aggrid_table(
        current_acct_df[["계좌 번호", "고객 번호", "계좌 유형", "상품코드", "개설일자", "평가적립금", "_account_id"]],
        key=GRID_KEYS["acct"],
        selection_mode="multiple",
        height=320,
        enable_filter=True,
        show_side_bar=False,
    )
    sel_acct = grid_acct.get("selected_rows", None)
    st.session_state.selected_accounts = get_all_values_from_selection(sel_acct, "_account_id")
    st.info(f"선택 계좌: {', '.join(map(str, st.session_state.selected_accounts)) or '없음'}")

    
    # -------- DC 계약 그리드 (계좌/고객 그리드 아래, 컨텍스트 미리보기 위) --------
    st.markdown("### ③ DC 계약")
    acct_ids = st.session_state.get("selected_accounts", [])
    try:
        dc_df = load_dc_contracts_from_db(acct_ids if acct_ids else None)
    except Exception:
        dc_df = pd.DataFrame()

    dc_view_cols = ["계약번호","근무처명","입사일자","중간정산일자","제도가입일자","부담금납입원금","운용손익금액","평가적립금합계금액","_ctrt_no"]
    cols = [c for c in dc_view_cols if c in getattr(dc_df, "columns", [])]
    if dc_df is not None and not dc_df.empty and cols:
        st.caption("DC 계약을 선택하세요 — 표시 컬럼: 계약번호, 근무처명, 입사일자, 중간정산일자, 제도가입일자, 부담금납입원금, 운용손익금액, 평가적립금합계금액")
        grid_dc = aggrid_table(
            dc_df[cols],
            key=GRID_KEYS["dc_ctrt"],
            selection_mode="single",
            height=280,
            enable_filter=True,
            show_side_bar=False,
        )
        sel_dc = grid_dc.get("selected_rows", None)
        st.session_state["selected_dc_row"] = (
            sel_dc[0] if isinstance(sel_dc, list) and len(sel_dc) > 0 else None
        )
    else:
        st.info("표시할 DC 계약 데이터가 없습니다.")
        st.session_state["selected_dc_row"] = None

    with st.expander("선택 컨텍스트 미리보기", expanded=False):
        ctx = build_context_for_chat()
        st.json(ctx)

    st.markdown('</div>', unsafe_allow_html=True)

with midsep:
    st.markdown('<div class="v-sep"></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-soft flush-top">', unsafe_allow_html=True)

    st.subheader("챗봇 · 시뮬레이션")

    # --- 디버그 토글 & KB 헬스체크 ---
    debug_on = st.toggle("디버그 모드", value=False, help="툴/RAG/이벤트/예외 로그 표시")
    if debug_on:
        ok, msg = kb_healthcheck()
        (st.success if ok else st.error)(f"KB Health: {msg}")

    # ---------- 파라미터 빌더 ----------
    def recalc_years_to_retirement():
        age = st.session_state.sim_params.get("customer_age")
        retire_age = st.session_state.sim_params.get("retirement_age", 65)
        if age is not None:
            st.session_state.sim_params["years_to_retirement"] = int(max(0, retire_age - age))

    def fill_from_selected_customer():
        cid = st.session_state.get("selected_customer")
        if not cid:
            return
        df = st.session_state.demo_df
        row = df[df["_customer_id"] == cid]
        if row.empty:
            return
        st.session_state.sim_params["customer_id"] = cid
        recalc_years_to_retirement()

    def fill_from_selected_accounts():
        df = st.session_state.acct_df
        sel = st.session_state.get("selected_accounts", [])
        if not sel:
            return
        bal = pd.to_numeric(df[df["_account_id"].isin(sel)]["평가적립금"], errors="coerce").fillna(0).sum()
        st.session_state.sim_params["current_balance"] = int(bal)

    def estimate_monthly_contribution_from_txn(window_days: int = 60):
        st.session_state.sim_params["monthly_contribution"] = 0

    with st.expander("연금 시뮬레이션 파라미터 빌더", expanded=True):
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            if st.button("1) 고객 정보 채우기", key=BUTTON_KEYS["fill_customer"]):
                fill_from_selected_customer()
        with colB:
            if st.button("2) 계좌 잔액 합산", key=BUTTON_KEYS["sum_accounts"]):
                fill_from_selected_accounts()
        with colC:
            if st.button("3) 최근입금→월납입 추정", key=BUTTON_KEYS["estimate_monthly"]):
                estimate_monthly_contribution_from_txn()

        p = st.session_state.sim_params
        st.markdown("**수동 조정**")
        col1, col2, col3 = st.columns(3)
        with col1:
            p["retirement_age"] = st.number_input("은퇴 나이", min_value=40, max_value=80,
                                                  value=int(p.get("retirement_age") or 65), step=1)
        with col2:
            p["expected_return_pct"] = st.number_input("연 수익률(%)", min_value=0.0, max_value=20.0,
                                                       value=float(p.get("expected_return_pct") or 4.0), step=0.1)
        with col3:
            p["inflation_pct"] = st.number_input("물가상승률(%)", min_value=0.0, max_value=10.0,
                                                 value=float(p.get("inflation_pct") or 2.0), step=0.1)
        recalc_years_to_retirement()
        p["notes"] = st.text_input("메모(선택)", value=p.get("notes") or "")

        st.caption("파라미터 미리보기(JSON)")
        # ▶ 시뮬레이션 유형을 선택하고, 왼쪽 컨텍스트를 합쳐 payload를 구성합니다.
        sim_type = st.selectbox("시뮬레이션 유형", ["dc_tax", "irp_recv", "annuity_recv"], index=0, key="sim_type_main")
        #payload_preview = {"cal_simul_type": sim_type, "params": p, "context": build_context_for_chat()}
        payload_preview = {
            "cal_simul_type": sim_type,
            "params": p,  # 내부용 영문 키
            "context": build_context_for_chat(),  # 내부용 영문 키 (customer/accounts/dc_contract)
        }
        # ▲ 보기는 한글 라벨, 내부는 영문 키 유지
        st.json(koreanize_payload(payload_preview))

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("초기화", type="secondary", key=BUTTON_KEYS["reset_params"]):
                st.session_state.sim_params = DEFAULT_PARAM_SCHEMA.copy()
        with c2:
            st.text_area("복사용 JSON", value=to_json_str(koreanize_payload(payload_preview)), height=160)
        with c3:
            if st.button("이 JSON을 챗봇에 전송", type="primary", key=BUTTON_KEYS["send_json"]):
                ctx = build_context_for_chat()
                user_json_prompt = f"다음 payload로 연금 시뮬레이션/상담을 수행해줘.\n\n```json\n{to_json_str(payload_preview)}\n```"
                # 하단 자유 질의 영역으로 전송을 위임합니다.
                st.session_state["queued_user_input"] = user_json_prompt
                st.session_state["queued_context"] = ctx
                try:
                    st.toast("전송됨: 하단 자유 질의 영역에서 응답합니다.", icon="✅")
                except Exception:
                    pass
                st.rerun()
        if debug_on:
            with st.expander("🔍 디버그 로그 (파라미터 전송)", expanded=False):
                dbg = st.session_state.last_debug
                if dbg.get("error"):
                    st.error(dbg["error"]["message"])
                    st.code(dbg["error"]["traceback"], language="python")
                st.caption(f"이벤트 수: {len(dbg.get('events', []))} | 총 소요: {dbg.get('timing',{}).get('total_sec','-')}s")
                st.json(dbg.get("events")[-50:] if dbg.get("events") else [])

        st.markdown("---")
        if st.button("🔧 툴로 바로 시뮬레이션 실행 (run_pension_simulator)", key=BUTTON_KEYS["run_tool"]):
            try:
                result = run_pension_simulator(p)
                st.success("툴 실행 완료")
                st.json(result)
                st.info(pretty_print_simulation(result))
            except Exception as e:
                st.error(f"툴 실행 중 에러: {e}")
                st.code(traceback.format_exc(), language="python")

    # ---------- 채팅 히스토리 & 일반 자유 질의 ----------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 히스토리 렌더
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        st.chat_message(role).markdown(msg["content"])

    # 공통 처리 함수: 사용자 프롬프트를 받아 스트리밍 응답까지 렌더
    def handle_chat_prompt(prompt: str, ctx: dict | None):
        if not prompt:
            return
        # 1) 히스토리 & 유저 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # 2) 컨텍스트 확정
        context = ctx or build_context_for_chat()

        # 3) 스트리밍 렌더
        resp_area = st.chat_message("assistant")
        placeholder = resp_area.empty()
        streamed = ""
        for chunk in run_agent_stream(prompt, context, debug=debug_on):
            streamed += chunk
            placeholder.markdown(streamed)

        # 4) 히스토리 저장
        st.session_state.messages.append({"role": "assistant", "content": streamed})

        # 5) 디버그
        if debug_on:
            with st.expander("🔍 디버그 로그 (채팅)", expanded=False):
                dbg = st.session_state.last_debug
                if dbg.get("error"):
                    st.error(dbg["error"]["message"])
                    st.code(dbg["error"]["traceback"], language="python")
                st.caption(f"이벤트 수: {len(dbg.get('events', []))} | 총 소요: {dbg.get('timing',{}).get('total_sec','-')}s")
                st.json(dbg.get("events")[-50:] if dbg.get("events") else [])

    # 1) 상단에서 큐잉된 요청(전송 버튼/이전 입력)이 있으면 먼저 처리
    queued = st.session_state.pop("queued_user_input", None)
    queued_ctx = st.session_state.pop("queued_context", None)
    if queued:
        handle_chat_prompt(queued, queued_ctx)

    # 2) 항상 맨 아래 입력창 표시
    st.markdown("---")  # 시각적 구분선 (선택)
    user_input = st.chat_input("질문을 입력하세요. (예: 선택 계좌 기반으로 은퇴 시 자산 추정해줘)")

    # 3) 입력이 들어온 프레임에서는 렌더하지 말고 큐에 넣고 즉시 rerun
    if user_input:
        st.session_state["queued_user_input"] = user_input
        st.session_state["queued_context"] = build_context_for_chat()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
