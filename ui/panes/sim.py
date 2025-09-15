# ui/panes/sim.py
import streamlit as st
import pandas as pd
import numpy as np

from typing import Any, Dict
from dataclasses import asdict, is_dataclass
from datetime import date as _date
from datetime import datetime

import math
from workspace.toolkits import pnsn_calculator

from agno.utils.log import logger
from ui.utils import _ctx_to_dict_any, update_ctx


def date_input_optional(label: str, *, default=None, key: str, help: str | None = None,
                        min_value=None, max_value=None):
    """
    checkbox로 '없음'을 선택하면 None을 반환.
    체크가 해제되면 date_input 값(date)을 반환.
    """
    c1, c2 = st.columns([4, 1])
    with c2:
        none_flag = st.checkbox("없음", key=f"{key}_none")
    with c1:
        dt = st.date_input(
            label,
            value=(default if default is not None else _date.today()),
            key=f"{key}_date",
            help=help,
            min_value=min_value,
            max_value=max_value,
            disabled=none_flag,
        )
    return None if none_flag else dt

def style_dataframe(df: pd.DataFrame):
    fmt = {}

    # float만 포맷팅
    for col in df.select_dtypes(include="float"):
        if col == "세율":
            fmt[col] = "{:.1%}"
        else:
            fmt[col] = "{:,.0f}"  # 소수 둘째자리까지 + 천 단위 구분

    # int만 포맷팅
    for col in df.select_dtypes(include="int"):
        fmt[col] = "{:,.0f}"

    return df.style.format(fmt)



def strptime_date_safe(val, fmt="%Y%m%d"):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (np.floating,)) and np.isnan(val):
        return None
    if not isinstance(val, str):
        return None

    s = val.strip()
    if s.lower() in {"", "nan", "na", "none", "null"}:
        return None

    try:
        return datetime.strptime(s, fmt).date()
    except ValueError:
        return None

def _save_df_to_context(dict_simul_result: Dict, df: pd.DataFrame, *, path=("sim_params"), key_name="df_capped"):
    """df를 records dict로 바꿔 context(sim_params)에 저장.
    """
    records = df.to_dict(orient="records")
    #logger.info(f"records: {records}")

    # TODO: 결과를 좀 더 예쁘게 정리해서 담을 필요가 있음
    # Agent 또는 Tool을 추가로 호출 
    update_ctx(
        sim_params={
            "산출내역": dict_simul_result,
            "산출내역 상세": records
        }
    )
    st.success("시뮬레이션 결과가 Context에 저장되었습니다.")


def render_sim_pane(ctx_obj: Any):

    ctx = _ctx_to_dict_any(ctx_obj)
    
    if not ctx:
        st.info(" 좌측에서 고객을 우선 선택하세요.")
        return

    st.markdown("##### 연금수령 시뮬레이션")
    accounts_display_info = [f"{x.get('acnt_type')} {x.get('account_id')}" for x in ctx.get("accounts") if x.get('acnt_type') in ['DC', 'IRP', '연금저축']]
    #logger.info(accounts_display_info)
    names = ["— 계좌를 선택하세요 —"] + accounts_display_info
    sel = st.selectbox("계좌 선택", options=names, index=0, key="sim_account_select_box")
    if sel == '— 계좌를 선택하세요 —':
        return
    else:
        # sel에서 필요한 항목 뽑아서 요약 만들기 (예시 파싱)
        _parts = sel.split(" ")
        _acc_no = _parts[1]
        _type = _parts[0] if len(_parts) > 1 else ""
        # 여기서 필요하면 추가 메타(개설일 등)를 DB/df에서 조회해 첨부
        st.caption(f"선택 계좌: **{_acc_no}** · 유형: **{_type}**")
        st.divider()
    
    sel_account = [x for x in ctx.get("accounts") if x.get('account_id') == sel.split(' ')[1]][0]

    cleaned_sel_account = {}

    date_fields = {k for k in sel_account.keys() if k.endswith("_dt")}
    amt_fields = {k for k in sel_account.keys() if k.endswith("_amt") or k.endswith("taxa")}
    for k, v in sel_account.items():
        # 1) 날짜 필드 처리
        if k in date_fields and v is None:
            cleaned_sel_account[k] = pd.to_datetime(_date.today(), format="%Y%m%d", errors="coerce").date()
            continue
        # 2) 금액 필드 처리
        if k in amt_fields:
            cleaned_sel_account[k] = int(0) if (pd.isna(v) or v is None) else int(v)
        else:
            cleaned_sel_account[k] = v

    # ★ 기본 디폴트 값 (요청값 반영)
    계좌번호_key = cleaned_sel_account.get("account_id") + cleaned_sel_account.get("prd_type_cd")
    평가기준일 = pd.to_datetime(_date.today(), format="%Y%m%d", errors="coerce").date()
    생년월일 = pd.to_datetime(ctx.get('customer_display').get("생년월일"), format="%Y%m%d", errors="coerce").date()
    입사일자 = pd.to_datetime(cleaned_sel_account.get("etco_dt", "19800101"), format="%Y%m%d", errors="coerce").date()
    퇴직일자 = pd.to_datetime(cleaned_sel_account.get("rtmt_dt", "19800101"), format="%Y%m%d", errors="coerce").date()

    # _def_IRP가입일   = _date(2014, 5, 1)
    제도가입일  = pd.to_datetime(cleaned_sel_account.get("acnt_bgn_dt", "19750101"), format="%Y%m%d", errors="coerce").date()
    # _def_연금개시일  = _date(2025, 9, 1)  # 퇴직일 이후
    #logger.info(f"date columns with {계좌번호_key} : {평가기준일} {생년월일} {입사일자} {퇴직일자} {제도가입일}")
    과세제외_자기부담금   = cleaned_sel_account.get("other_txtn_ecls_amt")
    이연퇴직소득         = cleaned_sel_account.get("user_almt_amt") + cleaned_sel_account.get("rtmt_incm_amt")
    세액공제자기부담금    = cleaned_sel_account.get("icdd_amt")
    운용손익             = cleaned_sel_account.get("acnt_evlu_amt") - cleaned_sel_account.get("other_txtn_ecls_amt") - cleaned_sel_account.get("user_almt_amt") - cleaned_sel_account.get("rtmt_incm_amt") - cleaned_sel_account.get("icdd_amt")
    이연퇴직소득세 = cleaned_sel_account.get("dfr_rtmt_taxa")
    #logger.info(f"amt columns : {과세제외_자기부담금} {이연퇴직소득} {세액공제자기부담금} {운용손익} {이연퇴직소득세}")

    운용수익률           = 0.03
    # ── ★ 계좌가 바뀌면 위젯 상태(키) 리셋 ───────────────────────────────
    base_key = f"{계좌번호_key}"  # 계좌별로 유니크
    prev_base_key = st.session_state.get("_prev_base_key")
    if prev_base_key and prev_base_key != base_key:
        # 일반 date_input 키
        for k in [
            "평가기준일",
            "생년월일",
            "제도가입일",
            "연금수령가능일",
            "연금개시일",
            "운용수익률",
        ]:
            st.session_state.pop(f"{k}_{prev_base_key}", None)
        # optional date_input 키(없음/날짜 위젯 둘 다)
        for k in ["입사일자", "퇴직일자", "제도가입일"]:
            st.session_state.pop(f"{k}_{prev_base_key}_none", None)
            st.session_state.pop(f"{k}_{prev_base_key}_date", None)
    st.session_state["_prev_base_key"] = base_key
    # ── 입력 폼 ─────────────────────────────────────────────────────────
    st.subheader("기본 정보(날짜)")
    d1, d2, d3 = st.columns(3)
    with d1:
        평가기준일 = st.date_input(
            "평가기준일", value=평가기준일, key=f"평가기준일_{base_key}"
        )
        생년월일 = st.date_input("생년월일", value=생년월일, key=f"생년월일_{base_key}")
        퇴직연금제도가입일 = st.date_input(
            "퇴직연금 제도가입일",
            value=제도가입일,
            key=f"제도가입일_{base_key}",
        )
    with d2:
        입사일자 = date_input_optional(
            "입사일자",
            default=입사일자,
            min_value=_date(1980, 1, 1),
            max_value=평가기준일,
            key=f"입사일_{base_key}",
            help="퇴직소득이 없으면 '없음' 체크",
        )

        퇴직일자 = date_input_optional(
            "퇴직일자",
            default=퇴직일자,
            min_value=_date(1980, 1, 1),
            max_value=_date(2050, 1, 1),
            key=f"퇴직일_{base_key}",
            help="퇴직소득이 없으면 '없음' 체크",
        )

    with d3:
        # ── 자동 산출 ────────────────────────────────────────────────────
        _연금수령가능일_dt = pnsn_calculator.calc_연금수령가능일(
            생년월일=생년월일, 제도가입일=제도가입일, 퇴직일자=퇴직일자
        )

        st.date_input(
            "연금수령가능일 (자동 계산)",
            value=_연금수령가능일_dt,
            disabled=True,
            help="퇴직일, 55세, 제도가입일+5년 중 가장 늦은 날",
            key=f"연금수령가능일_{base_key}",
        )
        연금개시일 = st.date_input(
            "연금개시일(연금수령가능일 이후)",
            value=max(_연금수령가능일_dt, 평가기준일),
            min_value=_연금수령가능일_dt,
            max_value=_date(2050, 1, 1),
            key=f"연금개시일_{base_key}",
        )
        운용수익률 = st.number_input(
            "연 운용수익률(예: 0.03=3%)",
            value=운용수익률,
            step=0.005,
            format="%.3f",
            key=f"운용수익률_{base_key}",
        )
    # ── 요약 표시 ────────────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    with b1:
        # 개시 나이(사용자 조정 가능)
        _auto_수령나이 = (연금개시일.year - 생년월일.year) - (
            1
            if (연금개시일.month, 연금개시일.day) < (생년월일.month, 생년월일.day)
            else 0
        )
        st.caption("연금개시 연령: " f"{_auto_수령나이}세")
    with b2:
        # 근속년수(사용자 조정 가능)
        if 퇴직일자 is not None and 입사일자 is not None:
            근속월수 = (퇴직일자.year - 입사일자.year) * 12 + (
                퇴직일자.month - 입사일자.month
            )
            if 퇴직일자.day < 입사일자.day:
                근속월수 -= 1
            _auto_근속년수 = math.ceil((근속월수 + 1) / 12)
        else:
            _auto_근속년수 = 0
        st.caption("근속년수: " f"{_auto_근속년수}년")
    with b3:
        _auto_연금수령연차 = max(0, 연금개시일.year - _연금수령가능일_dt.year) + (
            6 if 퇴직연금제도가입일 < _date(2013, 1, 1) else 1
        )
        st.caption("연금개시일 연금수령연차: " f"{_auto_연금수령연차}")

    st.subheader("연금소득 재원(원)")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        과세제외_자기부담금 = st.number_input(
            "과세제외 자기부담금", value=과세제외_자기부담금, step=100_000
        )
        st.caption(f"비과세: {int(과세제외_자기부담금):,} 원")
    with a2:
        이연퇴직소득 = st.number_input(
            "이연퇴직소득(= IRP 입금 퇴직금)", value=이연퇴직소득, step=1_000_000
        )
        st.caption(f"퇴직소득: {int(이연퇴직소득):,} 원")
    with a3:
        세액공제자기부담금 = st.number_input(
            "세액공제자기부담금", value=세액공제자기부담금, step=100_000
        )
        st.caption(f"세액공제: {int(세액공제자기부담금):,} 원")
    with a4:
        운용손익 = st.number_input("운용손익", value=운용손익, step=100_000)
        st.caption(f"운용손익: {int(운용손익):,} 원")
    # ── 우선순위: DB > 계산기 > 수동 ─────────────────────────────────────────
    if 이연퇴직소득세 is not None and 이연퇴직소득세 > 0:
        _default_tax_amt = int(이연퇴직소득세)
        _default_source = "db"
    else:
        _calc = pnsn_calculator.calc_퇴직소득세(
            근속년수=_auto_근속년수, 이연퇴직소득=이연퇴직소득
        )
        _default_tax_amt = int(_calc["퇴직소득산출세액"])
        _default_source = "calc"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_manual_tax_amount = st.checkbox(
            "퇴직소득세액 직접입력", key="use_manual_tax_amount"
        )

    # 수동 입력 기본값을 캡션 값과 동기화 (꺼져 있을 때는 항상 동기화)
    if "manual_tax_amount" not in st.session_state:
        st.session_state["manual_tax_amount"] = _default_tax_amt
    elif not st.session_state.get("use_manual_tax_amount", False):
        st.session_state["manual_tax_amount"] = _default_tax_amt

    with c2:
        manual_tax_amount = st.number_input(
            "퇴직소득 산출세액(원)",
            # value=st.session_state[
            #     "manual_tax_amount"
            # ],  # ← 캡션 값이 초기값으로 들어감
            step=1,
            key="manual_tax_amount",
            disabled=not use_manual_tax_amount,
            help="수동 입력을 켜면 편집할 수 있습니다.",
        )

    # 최종 적용 값 (우선순위: DB > 계산기 > 수동)
    if _default_source == "db":
        _tax_amt = _default_tax_amt
        source = "db"
    else:
        if not use_manual_tax_amount:
            _tax_amt = _default_tax_amt  # 계산기 값
            source = "calc"
        else:
            _tax_amt = int(st.session_state["manual_tax_amount"])
            source = "manual"

    _tax_rate = (
        (_tax_amt / 이연퇴직소득)
        if (이연퇴직소득 is not None and 이연퇴직소득 > 0)
        else 0.0
    )

    calc_퇴직소득세 = {
        "퇴직소득산출세액": _tax_amt,
        "퇴직소득세율": _tax_rate,
        "source": source,
    }

    # ── 경고: 계산기 산출 세액=0 이고 근속년수=0 인 경우(이연퇴직소득 > 0일 때만) ──
    if (
        source == "calc"  # 계산기 경로일 때만
        and _tax_amt == 0  # 계산기 결과 세액 0
        and 이연퇴직소득 is not None
        and 이연퇴직소득 > 0  # 과세 대상이 있을 때만 경고
        and _auto_근속년수 == 0  # 근속년수 입력 누락 추정
    ):
        missing = []
        if 입사일자 is None:
            missing.append("입사일")
        if 퇴직일자 is None:
            missing.append("퇴직일")

        if missing:
            st.warning(
                f"자동 산출 이연퇴직소득세가 0원이며 근속년수가 0년입니다. "
                f"{', '.join(missing)}을(를) 설정하세요.",
                icon="⚠️",
            )
        else:
            st.warning(
                "자동 산출 이연퇴직소득세가 0원이며 근속년수가 0년입니다. "
                "입사일/퇴직일을 확인하세요.",
                icon="⚠️",
            )

    st.caption(
        f"총평가금액(= 과세제외 자기부담금 + 이연퇴직소득 + 세액공제자기부담금 + 운용손익)): "
        f"{과세제외_자기부담금 + 이연퇴직소득 + 세액공제자기부담금 + 운용손익:,} 원"
    )
    label = {"db": "Wink", "calc": "자동계산", "manual": "직접입력"}
    st.caption(f"퇴직소득세율({label[source]}): {calc_퇴직소득세['퇴직소득세율']:.1%}")

    st.caption(
        f"퇴직소득 산출세액({label[source]}): {calc_퇴직소득세['퇴직소득산출세액']:,} 원"
    )

    st.subheader("지급 옵션")
    c1, c2, c3 = st.columns(3)
    with c1:
        지급옵션 = st.selectbox(
            "지급옵션",
            ["기간확정형", "금액확정형", "한도수령", "최소수령", "일시금"],
            index=0,
            key="지급옵션",
        )

    if 지급옵션 == "기간확정형":
        with c2:
            지급기간_년 = st.number_input(
                "지급기간_년(필수)",
                min_value=1,
                value=5,
                step=1,
            )
        수령금액_년 = None

    elif 지급옵션 == "금액확정형":
        with c2:
            수령금액_년 = st.number_input(
                "수령금액_년(필수, 원)", min_value=1, value=12_000_000, step=100_000
            )
        지급기간_년 = None

    else:
        # 한도수령, 최소수령일 경우
        지급기간_년, 수령금액_년 = None, None

    submitted_option = st.button("시뮬레이션 실행")    
    
    if submitted_option:
        params = dict(
            평가기준일=평가기준일,
            # ↓ pnsn_sim_calculator.simulate_pension이 '연금수령가능일'을 직접 쓰는 구조라면 이 값을 사용
            연금개시일=연금개시일,
            생년월일=생년월일,
            입사일자=입사일자,
            퇴직일자=퇴직일자,
            퇴직연금제도가입일=퇴직연금제도가입일,
            운용수익률=float(운용수익률),
            과세제외_자기부담금=int(과세제외_자기부담금),
            이연퇴직소득=int(이연퇴직소득),
            그외=int(세액공제자기부담금 + 운용손익),
            퇴직소득산출세액=calc_퇴직소득세["퇴직소득산출세액"],
            지급옵션=지급옵션,
            지급기간_년=int(지급기간_년) if 지급기간_년 else None,
            수령금액_년=int(수령금액_년) if 수령금액_년 else None,
        )
        if use_manual_tax_amount:
            params["퇴직소득산출세액_직접입력"] = int(manual_tax_amount)
        # 필수 검증
        if 지급옵션 == "기간확정형" and not params["지급기간_년"]:
            st.error("기간확정형에는 '지급기간_년'이 필요합니다.")
            st.stop()
        if 지급옵션 == "금액확정형" and not params["수령금액_년"]:
            st.error("금액확정형에는 '수령금액_년'이 필요합니다.")
            st.stop()

        try:
            with st.spinner("계산 중..."):
                df_capped = pnsn_calculator.simulate_pension(**params)
                # 일시금 지급옵션 추가 계산
                params_lump = params.copy()
                params_lump["지급옵션"] = "일시금"
                df_lump = pnsn_calculator.simulate_pension(**params_lump)

            dict_simul_result=dict()
            # 입력값 요약 + 결과 출력
            with st.container(border=True):
                st.markdown("##### 산출결과")
                dict_simul_result['연금개시정보'] = dict()
                m1, m2, m3, m4 = st.columns(4)
                _auto_현재나이 = (평가기준일.year - 생년월일.year) - (
                    1
                    if (연금개시일.month, 연금개시일.day)
                    < (생년월일.month, 생년월일.day)
                    else 0
                )
                with m1:
                    dict_simul_result['연금개시정보']['현재연령'] = _auto_현재나이
                    st.metric("현재연령", f"{_auto_현재나이} 세")
                with m2:
                    dict_simul_result['연금개시정보']['연금개시일자'] = 연금개시일
                    st.metric("연금개시일자", f"{연금개시일}")
                with m3:
                    dict_simul_result['연금개시정보']['연금개시연령'] = _auto_수령나이
                    st.metric("연금개시연령", f"{_auto_수령나이}세")
                with m4:
                    dict_simul_result['연금개시정보']['연금개시금액'] = f"{int(df_capped[df_capped['지급회차']==1]['지급전잔액'].values[0]):,}"
                    st.metric(
                        "연금개시금액",
                        f"{int(df_capped[df_capped['지급회차']==1]['지급전잔액'].values[0]):,} 원",
                    )
                # 지급 옵션별 금액
                dict_simul_result['연금수령정보'] = dict()
                dict_simul_result['연금수령정보'][지급옵션] = dict()
                if {"총세액", "실수령액", "실제지급액"}.issubset(df_capped.columns):
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        dict_simul_result['연금수령정보'][지급옵션]['총 연금수령액'] =f"{int(df_capped['실제지급액'].sum()):,}"
                        st.metric(
                            "총 연금수령액",
                            f"{int(df_capped['실제지급액'].sum()):,} 원",
                        )
                    with m2:
                        dict_simul_result['연금수령정보'][지급옵션]['총 세액 합계'] =f"{int(df_capped['총세액'].sum()):,}"
                        st.metric(
                            "총 세액 합계", f"{int(df_capped['총세액'].sum()):,} 원"
                        )
                    with m3:
                        dict_simul_result['연금수령정보'][지급옵션]['실수령 합계'] =f"{int(df_capped['실수령액'].sum()):,}"
                        st.metric(
                            "실수령 합계", f"{int(df_capped['실수령액'].sum()):,} 원"
                        )
                    eff_tax_rate = (
                        df_capped["총세액"].sum() / df_capped["실제지급액"].sum()
                        if df_capped["실제지급액"].sum() > 0
                        else 0
                    )
                    with m4:
                        dict_simul_result['연금수령정보'][지급옵션]['실효세율'] =f"{eff_tax_rate:.1%}"
                        st.metric("실효세율", f"{eff_tax_rate:.1%}")

            with st.container(border=True):
                dict_simul_result['연금수령정보']['일시금'] = dict()
                st.markdown("##### (일시금 수령 시)")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    dict_simul_result['연금수령정보']['일시금']['총 연금수령액'] = f"{int(df_lump['실제지급액'].sum()):,}"
                    st.metric(
                        "총 연금수령액", f"{int(df_lump['실제지급액'].sum()):,} 원"
                    )
                with m2:
                    dict_simul_result['연금수령정보']['일시금']['총 세액 합계'] = f"{int(df_lump['총세액'].sum()):,}"
                    st.metric("총 세액 합계", f"{int(df_lump['총세액'].sum()):,} 원")
                with m3:
                    dict_simul_result['연금수령정보']['일시금']['실수령 합계'] = f"{int(df_lump['실수령액'].sum()):,}"
                    st.metric("실수령 합계", f"{int(df_lump['실수령액'].sum()):,} 원")
                eff_tax_rate_lump = (
                    df_lump["총세액"].sum() / df_lump["실제지급액"].sum()
                    if df_lump["실제지급액"].sum() > 0
                    else 0
                )
                with m4:
                    dict_simul_result['연금수령정보']['일시금']['실효세율'] = f"{eff_tax_rate_lump:.1%}"
                    st.metric("실효세율", f"{eff_tax_rate_lump:.1%}")

            st.markdown("##### 산출결과 내역")
            # 1) 컬럼 생성
            df_capped["한도초과여부"] = df_capped.apply(
                lambda x: (
                    "한도 이내"
                    if pd.isna(x["한도"]) or x["한도"] >= x["실제지급액"]
                    else "한도 초과"
                ),
                axis=1,
            )

            # 2) 스타일 적용 (DataFrame 먼저 자른 후 .style 사용)
            col_view = [
                "지급회차",
                "나이",
                "지급전잔액",
                "한도",
                "실제지급액",
                "총세액",
                "실수령액",
                "세율",
                "지급옵션",
                "한도초과여부",
            ]

            styled_df = style_dataframe(df_capped[col_view]).map(
                lambda v: "color:green;" if v == "한도 이내" else "color:red;",
                subset=["한도초과여부"],
            )

            # 3) 출력
            st.dataframe(styled_df, width="stretch", hide_index=True)

            st.markdown("##### 산출결과 세부내역")
            st.dataframe(
                style_dataframe(df_capped),
                column_config={
                    "연금지급일": st.column_config.DateColumn(
                        "연금지급일", format="YYYY-MM-DD"
                    ),
                    "과세기간개시일": st.column_config.DateColumn(
                        "과세기간개시일", format="YYYY-MM-DD"
                    ),
                },
                width="stretch",
                hide_index=True,
            )

            logger.info(f"df_capped: {df_capped}")
            # --- CSV와 컨텍스트 저장 버튼을 같은 레벨 + 크게 ---
            btn1, btn2 = st.columns([1, 1])

            with btn1:
                st.download_button(
                    "📥 CSV 다운로드",
                    data=df_capped.to_csv(index=False).encode("utf-8-sig"),
                    file_name="연금시뮬레이션_df_capped.csv",
                    mime="text/csv",
                    key="btn_csv_download",
                    width="stretch"
                )

            with btn2:
                st.button(
                    "💾 컨텍스트에 저장", 
                    on_click=_save_df_to_context,
                    args = (dict_simul_result, df_capped),
                    key="btn_save_to_context", 
                    width="stretch")

        except Exception as e:
            st.error("시뮬레이션 중 오류가 발생했습니다.")
            st.exception(e)
