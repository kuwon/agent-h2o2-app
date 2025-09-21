from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import streamlit as st
from datetime import datetime, date

from ui.components.policy_engine import load_policies, evaluate_policies

MARKDOWN_DIR = Path("resources/markdown")  # 필요 시 경로 조정

# ─────────────────────────────────────────────
# Helpers
# Safe evaluate wrapper & fallback evaluator
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
    d = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if callable(val):
            continue
        d[attr] = val
    return d

def _normalize_id(x: Any) -> str:
    try:
        s = str(x).strip()
        return s
    except Exception:
        return ""

def _get_snippets_map(item: Any) -> Dict[str, str]:
    """
    policy 결과 객체/딕셔너리에서 snippets 맵을 최대한 끌어옴.
    - item.snippets (dict)
    - item._raw.snippets / item.source.snippets 등 잠재적 내부 보관소도 탐색
    - item 자체가 dict면 평탄화 후 'snippets' 키 우선
    """
    # 1차: 직접 접근
    cand = _sg(item, "snippets")
    if isinstance(cand, dict):
        # key를 문자열화
        return { _normalize_id(k): str(v) for k, v in cand.items() if v is not None }

    # 2차: 내부 보관소 비슷한 것들 시도
    for k in ("raw", "_raw", "source", "_source", "policy", "_policy"):
        inner = _sg(item, k)
        if inner is None:
            continue
        # inner가 객체/딕셔너리일 때 그 안의 snippets
        inner_snips = _sg(inner, "snippets")
        if isinstance(inner_snips, dict):
            return { _normalize_id(k): str(v) for k, v in inner_snips.items() if v is not None }

    # 3차: 평탄화해서 찾기
    d = _as_dict(item)
    sn = d.get("snippets")
    if isinstance(sn, dict):
        return { _normalize_id(k): str(v) for k, v in sn.items() if v is not None }

    return {}

def _try_evaluate_policies(policies, customer_row, accounts_df):
    """
    1차: 기존 evaluate_policies 사용
    2차: AttributeError('... .get') 등 발생 시 객체→dict 평탄화 후 폴백 평가기로 처리
    """
    try:
        return evaluate_policies(policies, customer_row, accounts_df)
    except AttributeError:
        pol_dicts = [_as_dict(p) for p in (policies or [])]
        return _evaluate_policies_fallback(pol_dicts, customer_row, accounts_df)
    except Exception:
        raise

def _apply_where_df(df: pd.DataFrame, where: dict | None) -> pd.DataFrame:
    if where is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for k, v in where.items():
        if k.endswith("_in"):
            col = k[:-3]
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            if col in df.columns:
                mask &= df[col].isin(list(vals))
        else:
            col = k
            if col in df.columns:
                mask &= (df[col] == v)
    return df[mask]

def _to_date(x) -> date | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (datetime, date)):
        return x.date() if isinstance(x, datetime) else x
    s = str(x).strip()
    if not s:
        return None
    try:
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").date()
        return datetime.fromisoformat(s).date()
    except Exception:
        return None

def _days_until(d: date | None, today: date | None = None) -> int | None:
    if d is None:
        return None
    t = today or date.today()
    return (d - t).days

def _eval_cond_fallback(cond: dict, customer_row: pd.Series, accounts_df: pd.DataFrame) -> dict:
    cid   = cond.get("id") or cond.get("name")
    field = str(cond.get("field") or "")
    op    = str(cond.get("op") or "")
    value = cond.get("value")
    where = cond.get("where")
    any_all = str(cond.get("any_all") or "any").lower()

    parts = field.split(".")
    ns = parts[0] if parts else ""
    col = parts[1] if len(parts) > 1 else ""
    current, result = None, False

    if ns == "accounts":
        sub = _apply_where_df(accounts_df, where)
        if op.startswith("sum_"):
            s = float(pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sub.empty else 0.0
            current = s
            if isinstance(value, (int, float)):
                if   op == "sum_gte": result = s >= value
                elif op == "sum_gt":  result = s >  value
                elif op == "sum_lte": result = s <= value
                elif op == "sum_lt":  result = s <  value
                elif op == "sum_eq":  result = s == value
                elif op == "sum_ne":  result = s != value
        elif op.startswith("count_"):
            c = len(sub) if (col not in sub.columns) else sub[col].notna().sum()
            current = int(c)
            if isinstance(value, (int, float)):
                v = float(value)
                if   op == "count_gte": result = c >= v
                elif op == "count_gt":  result = c >  v
                elif op == "count_lte": result = c <= v
                elif op == "count_lt":  result = c <  v
                elif op == "count_eq":  result = c == v
                elif op == "count_ne":  result = c != v
        elif op == "within_days":
            flags = []
            dds = []
            for _, r in sub.iterrows():
                dd = _days_until(_to_date(r.get(col)))
                dds.append(dd)
                flags.append(dd is not None and abs(dd) <= int(value))
            current = dds
            result = (all(flags) if any_all == "all" else any(flags))
        elif op in ("exists", "not_exists"):
            current = f"rows={len(sub)}"
            ex = not sub.empty
            result = (ex if op == "exists" else not ex)
        else:
            current, result = "(unsupported op)", False

    elif ns in ("customer", "customers"):
        cur = customer_row.get(col)
        current = cur
        if op in ("gte","gt","lte","lt","eq","ne"):
            try:
                cnum = float(cur); vnum = float(value)
                if   op == "gte": result = cnum >= vnum
                elif op == "gt":  result = cnum >  vnum
                elif op == "lte": result = cnum <= vnum
                elif op == "lt":  result = cnum <  vnum
                elif op == "eq":  result = cnum == vnum
                elif op == "ne":  result = cnum != vnum
            except Exception:
                if   op == "eq": result = str(cur) == str(value)
                elif op == "ne": result = str(cur) != str(value)
                else: result = False
        elif op == "within_days":
            dd = _days_until(_to_date(cur))
            current = dd
            result = (dd is not None and abs(dd) <= int(value))
        else:
            current, result = "(unsupported op)", False
    else:
        current, result = "(unknown namespace)", False

    # cond-level snippet만 보유. 상위 policy.snippets는 UI에서 별도로 합쳐서 조회
    return {
        "id": cid, "field": field, "op": op, "value": value,
        "current": current, "result": result, "where": where,
        "snippet": cond.get("snippet")
    }

def _evaluate_policies_fallback(policies: list, customer_row: pd.Series, accounts_df: pd.DataFrame) -> list:
    out = []
    for pol in (policies or []):
        p = pol if isinstance(pol, dict) else _as_dict(pol)
        conds = p.get("conditions", []) or []
        eval_conds = []
        for c in conds:
            cdict = c if isinstance(c, dict) else _as_dict(c)
            eval_conds.append(_eval_cond_fallback(cdict, customer_row, accounts_df))
        out.append({
            "policy_id": p.get("id") or p.get("policy_id"),
            "title": p.get("title"),
            "anchor": p.get("anchor"),
            "file": p.get("_file") or p.get("file"),
            "conditions": eval_conds,
            "effects": p.get("effects", {}) or {},
            "snippets": p.get("snippets", {}) or {},
        })
    return out

def _format_current(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if isinstance(val, (list, tuple)):
        try:
            return ", ".join("-" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v) for v in val)
        except Exception:
            return str(val)
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
    if not customer:
        return False
    cid = str(customer.get("customer_id") or "").strip()
    return cid != ""

# 사람이 읽기 쉬운 조건 요약 만들기 (snippet 폴백)
_OP_MAP = {
    "sum_lt": "합계가 {v:,}원 미만",
    "sum_lte": "합계가 {v:,}원 이하",
    "sum_gte": "합계가 {v:,}원 이상",
    "sum_gt": "합계가 {v:,}원 초과",
    "sum_eq": "합계가 {v:,}원과 같음",
    "sum_ne": "합계가 {v:,}원과 다름",
    "count_gte": "개수가 {v}개 이상",
    "count_gt": "개수가 {v}개 초과",
    "count_lte": "개수가 {v}개 이하",
    "count_lt": "개수가 {v}개 미만",
    "count_eq": "개수가 {v}개",
    "count_ne": "개수가 {v}개 아님",
    "within_days": "{v}일 이내",
    "exists": "해당되는 계좌 존재",
    "not_exists": "해당되는 계좌 없음",
    "gte": "{v} 이상", "gt": "{v} 초과", "lte": "{v} 이하", "lt": "{v} 미만",
    "eq": "{v}와 같음", "ne": "{v}와 다름",
}

def _humanize_where(where: dict | None) -> str:
    if not isinstance(where, dict) or not where:
        return ""
    parts = []
    for k, v in where.items():
        key = k.replace("_in", "")
        if isinstance(v, (list, tuple, set)):
            parts.append(f"{key}: {', '.join(map(str, v))}")
        else:
            parts.append(f"{key}: {v}")
    return " (범위: " + " / ".join(parts) + ")"

def _humanize_condition(cond: Any) -> str:
    """cond를 사람이 읽기 쉬운 한 줄 요약으로 변환(정책 스니펫 폴백)."""
    cid   = _sg(cond, "id") or _sg(cond, "name") or ""
    field = _sg(cond, "field") or ""
    op    = _sg(cond, "op") or ""
    val   = _sg(cond, "value")
    where = _sg(cond, "where")
    # 필드 요약
    if field.startswith("accounts."):
        target = "계좌 납입합계" if field.endswith("copt_year_pymt_amt") else field.split(".", 1)[1]
    elif field.startswith("customer") or field.startswith("customers."):
        target = "고객 정보 " + field.split(".", 1)[1]
    else:
        target = field or "조건"
    op_fmt = _OP_MAP.get(op, op)
    try:
        op_txt = op_fmt.format(v=int(val) if isinstance(val, (int, float)) else val)
    except Exception:
        op_txt = op_fmt
    scope = _humanize_where(where)
    #return f"{cid} — {target}: {op_txt}{scope}"
    return f"{op_txt}{scope}"

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def render_policy_adaption_section(
    customer: Dict[str, Any],
    accounts: List[Dict[str, Any]],
    *,
    max_policies: int = 3,
) -> None:
    """정책 적용 섹션 렌더러 (Dict 입력)"""
    if not _has_selected_customer(customer):
        st.info("좌측에서 고객을 선택하면 정책 적용 결과를 보여드릴게요.")
        return

    customer_row, accounts_df = _coerce_inputs(customer, accounts)

    try:
        policies = load_policies(MARKDOWN_DIR)
    except Exception as ex:
        st.warning(f"정책(.md) 로딩 오류: {ex}")
        return
    if not policies:
        st.info("정책 .md에서 policy 블록을 찾지 못했습니다. (```yaml policy: ... ``` 형태)")
        return

    try:
        evaled = _try_evaluate_policies(policies, customer_row, accounts_df)
    except Exception as ex:
        st.warning(f"정책 판정 중 오류: {ex}")
        return

    items = evaled[:max_policies] if len(evaled) > max_policies else evaled

    for item in items:
        title     = _sg(item, "title") or _sg(item, "policy_id") or _sg(item, "id") or "정책"
        anchor    = _sg(item, "anchor")
        file_path = _sg(item, "file") or _sg(item, "_file")
        conditions = _sg(item, "conditions") or []

        with st.container(border=True):
            header = f"**{title}**"
            if anchor:
                header += f"  \n*근거 위치: `{anchor}`*"
            st.markdown(header)

            # 적용 범위 배지(where)
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

            # ───────── 근거 요약(스니펫) ─────────
            snip_map = _get_snippets_map(item)
            pos_snips, other_snips = [], []
            for c in conditions:
                cid = _normalize_id(_sg(c, "id") or _sg(c, "name"))
                # 우선순위: cond.snippet → policy.snippets[cid] → 사람이 읽는 폴백
                snip = _sg(c, "snippet")
                if not snip and cid:
                    snip = snip_map.get(cid)
                if not snip:
                    snip = _humanize_condition(c)
                bucket = pos_snips if bool(_sg(c, "result", False)) else other_snips
                bucket.append((cid or "-", snip))

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

            # ───────── 계산/판정 상세(Expander) ─────────
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
                    st.dataframe(df, width="stretch", hide_index=True)
                else:
                    st.caption("표시할 조건이 없습니다.")

    if len(evaled) > len(items):
        st.link_button("더 많은 정책 결과 보기", "javascript:window.scrollTo(0,0);")
