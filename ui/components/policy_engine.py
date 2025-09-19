# ui/policy_engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import yaml
import pandas as pd
from datetime import date, datetime, timedelta

# -------------------------
# Data structures
# -------------------------
@dataclass
class Condition:
    id: str
    field: str               # e.g., 'customers.brth_dt' or 'accounts.expd_dt'
    op: str                  # e.g., 'age_gte_years', 'within_days', ...
    value: Optional[Union[int, float, str]] = None
    any_all: str = "any"     # 'any'|'all' for accounts.* aggregation (default any)

@dataclass
class Policy:
    pid: str
    title: str
    file_path: Path
    anchor: Optional[str]
    conditions: List[Condition]
    effects: Dict[str, List[str]]  # e.g., {'eligible': [...], 'caution': [...], 'info': [...]}
    snippet_map: Dict[str, str]    # condition_id -> markdown snippet (optional)


# -------------------------
# Utilities
# -------------------------
def _to_date(x: Any) -> Optional[date]:
    if x is None or pd.isna(x):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _age_in_years(birth: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if birth is None:
        return None
    today = today or date.today()
    # basic age calc
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

def _days_until(d: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if d is None:
        return None
    today = today or date.today()
    return (d - today).days

def _months_since(d: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if d is None:
        return None
    today = today or date.today()
    return (today.year - d.year) * 12 + (today.month - d.month)

def _safe_sum(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())

# -------------------------
# Markdown → Policy parsing
# -------------------------
POLICY_BLOCK_PATTERN = re.compile(
    r"```(?:yaml|yml)\s+policy:(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)

def _extract_policy_yaml_blocks(md_text: str) -> List[Dict[str, Any]]:
    """
    From a markdown string, extract fenced code blocks like:

    ```yaml policy:
    id: pension_maturity
    title: 만기 임박 고객 안내
    anchor: "#maturity"
    conditions:
      - id: maturity_within_90
        field: accounts.expd_dt
        op: within_days
        value: 90
        any_all: any
    effects:
      eligible: []
      caution: [maturity_within_90]
      info: []
    snippets:
      maturity_within_90: >
        만기가 90일 이내인 경우 △△절차를 안내합니다.
    ```

    Returns list of dicts (one md can contain multiple policy blocks).
    """
    blocks = []
    for m in POLICY_BLOCK_PATTERN.finditer(md_text):
        raw = m.group(1)
        try:
            data = yaml.safe_load(raw) or {}
            blocks.append(data)
        except Exception:
            # Ignore malformed blocks gracefully
            pass
    return blocks

def load_policies(markdown_dir: Union[str, Path]) -> List[Policy]:
    md_dir = Path(markdown_dir)
    policies: List[Policy] = []
    for p in sorted(md_dir.glob("**/*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for block in _extract_policy_yaml_blocks(text):
            pid = str(block.get("id") or p.stem)
            title = str(block.get("title") or p.stem)
            anchor = block.get("anchor")
            conds_raw = block.get("conditions", []) or []
            effects = block.get("effects", {}) or {}
            snippets = block.get("snippets", {}) or {}

            conditions: List[Condition] = []
            for c in conds_raw:
                conditions.append(
                    Condition(
                        id=str(c.get("id")),
                        field=str(c.get("field")),
                        op=str(c.get("op")),
                        value=c.get("value"),
                        any_all=str(c.get("any_all", "any")).lower(),
                    )
                )

            policies.append(
                Policy(
                    pid=pid,
                    title=title,
                    file_path=p,
                    anchor=anchor,
                    conditions=conditions,
                    effects=effects,
                    snippet_map={str(k): str(v) for k, v in snippets.items()},
                )
            )
    return policies


# -------------------------
# Condition evaluation
# -------------------------
def _resolve_field(
    field: str,
    customer_row: pd.Series,
    accounts_df: pd.DataFrame
) -> Union[Any, pd.Series, None]:
    """
    field can be 'customers.xxx' or 'accounts.yyy'
    Return single value for customers.*, and a Series for accounts.*.
    """
    if field.startswith("customers."):
        key = field.split(".", 1)[1]
        return customer_row.get(key)
    elif field.startswith("accounts."):
        key = field.split(".", 1)[1]
        if key in accounts_df.columns:
            return accounts_df[key]
        return pd.Series([], dtype="object")
    return None

def _eval_op_on_accounts_series(
    series: pd.Series, op: str, value: Any
) -> Tuple[bool, Optional[Any]]:
    """
    Evaluate condition against accounts.* series.
    Returns (bool_result, representative_value_for_display)
    """
    today = date.today()

    if op in ("is_null", "is_not_null"):
        mask = series.isna()
        return ((mask.all() if op == "is_null" else (~mask).any()), None)

    if op == "within_days":
        # True if any account date is within 'value' days from today (future or past?)
        days = int(value)
        dates = series.map(_to_date)
        in_window = dates.map(lambda d: (abs(_days_until(d, today)) if d else None))
        ok = in_window.dropna().map(lambda dd: dd <= days).any()
        # representative min days-to
        rep = None
        if not dates.dropna().empty:
            rep = min((abs(_days_until(d, today)) for d in dates if d), default=None)
        return ok, rep

    if op == "tenor_gte_months":
        # months since account begin date
        dates = series.map(_to_date)
        months = dates.map(lambda d: _months_since(d, today) if d else None)
        ok = (months.dropna() >= int(value)).any()
        rep = None if months.dropna().empty else int(months.dropna().max())
        return ok, rep

    if op == "sum_gte":
        s = _safe_sum(series)
        return (s >= float(value)), s

    if op == "sum_ratio_gte":
        # value expects dict: {numerator: 'accounts.user_almt_amt', denominator: 'accounts.sbsr_almt_amt', threshold: 0.5}
        thr = float(value.get("threshold", 0))
        # NOTE: series here is unused; we recompute from provided fields
        return None, None  # handled upstream (we won't hit here directly)

    # fallback: try equality
    try:
        return (series.eq(value).any(), None)
    except Exception:
        return False, None


def _eval_condition(
    cond: Condition,
    customer_row: pd.Series,
    accounts_df: pd.DataFrame
) -> Tuple[bool, Optional[Any]]:
    """
    Returns (result_bool, representative_value_for_display)
    representative_value helps show "현재값" 칸에 무엇을 보여줄지 결정
    """
    field_val = _resolve_field(cond.field, customer_row, accounts_df)
    today = date.today()

    # Customers.* (scalar)
    if cond.field.startswith("customers."):
        if cond.op == "is_null":
            return (pd.isna(field_val), field_val)
        if cond.op == "is_not_null":
            return (not pd.isna(field_val), field_val)
        if cond.op == "age_gte_years":
            birth = _to_date(field_val)
            age = _age_in_years(birth, today)
            return ((age is not None and age >= int(cond.value)), age)
        if cond.op == "age_lt_years":
            birth = _to_date(field_val)
            age = _age_in_years(birth, today)
            return ((age is not None and age < int(cond.value)), age)
        # equality fallback
        return (field_val == cond.value, field_val)

    # Accounts.* (Series)
    if cond.field.startswith("accounts."):
        s = field_val if isinstance(field_val, pd.Series) else pd.Series([], dtype="object")

        if cond.op == "sum_ratio_gte":
            # Need numerator/denominator fields
            spec = cond.value or {}
            num_field = spec.get("numerator")
            den_field = spec.get("denominator")
            thr = float(spec.get("threshold", 0))
            num = _resolve_field(num_field, customer_row, accounts_df)
            den = _resolve_field(den_field, customer_row, accounts_df)
            num_sum = _safe_sum(pd.Series(num)) if isinstance(num, pd.Series) else 0
            den_sum = _safe_sum(pd.Series(den)) if isinstance(den, pd.Series) else 0
            ratio = None
            if den_sum != 0:
                ratio = num_sum / den_sum
            return ((ratio is not None and ratio >= thr), ratio)

        # generic accounts ops
        ok, rep = _eval_op_on_accounts_series(s, cond.op, cond.value)
        if cond.any_all == "all" and cond.op in ("is_null", "is_not_null"):
            # Already handled by _eval_op_on_accounts_series for is_(not_)null
            return ok, rep
        # For other ops, "any" semantics are already default
        return ok, rep

    return False, None


def evaluate_policies(
    policies: List[Policy],
    customer_row: pd.Series,
    accounts_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dict:
    {
      'policy_id': ...,
      'title': ...,
      'file': ...,
      'anchor': ...,
      'effects': {'eligible': [...], 'caution': [...], 'info': [...]},
      'conditions': [
          {'id': '...', 'field': 'accounts.expd_dt', 'op': 'within_days', 'value': 90,
           'result': True, 'current': 14, 'snippet': '...'},
          ...
      ],
      'score': 2  # number of positive conditions, used for prioritization
    }
    """
    results = []
    for pol in policies:
        cond_rows = []
        positives = 0
        for c in pol.conditions:
            ok, rep = _eval_condition(c, customer_row, accounts_df)
            if ok:
                positives += 1
            cond_rows.append({
                "id": c.id,
                "field": c.field,
                "op": c.op,
                "value": c.value,
                "any_all": c.any_all,
                "result": bool(ok),
                "current": rep,
                "snippet": pol.snippet_map.get(c.id)
            })
        results.append({
            "policy_id": pol.pid,
            "title": pol.title,
            "file": str(pol.file_path),
            "anchor": pol.anchor,
            "effects": pol.effects,
            "conditions": cond_rows,
            "score": positives
        })
    # sort by score desc
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# -------------------------
# Timeline builder
# -------------------------
def build_timeline(
    customer_row: pd.Series,
    accounts_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Returns event list sorted by date:
    { 'date': date, 'label': '계좌 개설', 'kind': 'begin', 'meta': {...}, 'd_day': 'D-30' }
    """
    events: List[Dict[str, Any]] = []
    today = date.today()

    # Customer-level
    brth = _to_date(customer_row.get("brth_dt"))
    if brth:
        events.append({"date": brth, "label": "생년월일", "kind": "birth", "meta": {}, "d_day": None})

    # Accounts-level (iterate rows)
    for _, row in accounts_df.iterrows():
        aid = row.get("account_id")
        atype = row.get("acnt_type")  # ← 추가(보조 정보)
        ab = _to_date(row.get("acnt_bgn_dt"))
        if ab:
            events.append({
                "date": ab, "label": "계좌 개설", "kind": "begin",
                "meta": {"account_id": aid, "acnt_type": atype},  # ← 추가
                "d_day": None
            })
        md = _to_date(row.get("midl_excc_dt"))
        if md:
            events.append({
                "date": md, "label": "중간정산", "kind": "mid",
                "meta": {"account_id": aid, "acnt_type": atype},  # ← 추가
                "d_day": None
            })
        rt = _to_date(row.get("rtmt_dt"))
        if rt:
            events.append({
                "date": rt, "label": "퇴직", "kind": "retire",
                "meta": {"account_id": aid, "acnt_type": atype},  # ← 추가
                "d_day": None
            })
        ex = _to_date(row.get("expd_dt"))
        if ex:
            dd = _days_until(ex, today)
            dd_str = None if dd is None else (f"D-{dd}" if dd >= 0 else f"D+{abs(dd)}")
            events.append({
                "date": ex, "label": "만기", "kind": "maturity",
                "meta": {"account_id": aid, "acnt_type": atype},  # ← 추가
                "d_day": dd_str
            })

    # sort
    events.sort(key=lambda e: (e["date"] is None, e["date"]))
    return events

