# ui/panes/sim.py
import streamlit as st
import pandas as pd
from typing import Any, Dict
from dataclasses import asdict, is_dataclass
from datetime import date as _date
import math
from workspace.toolkits import pnsn_calculator # import (_to_date, 절사10원, add_year_safe, calc_근속년수공제, calc_환산급여, calc_환산급여별공제, calc_환산산출세액, calc_퇴직소득세, calc_연금수령가능일, simulate_pension)


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


def _ctx_to_dict(ctx: Any) -> Dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        return ctx
    if is_dataclass(ctx):
        try:
            return asdict(ctx)
        except Exception:
            return getattr(ctx, "__dict__", {}) or {}
    return getattr(ctx, "__dict__", {}) or {}

def render_sim_pane(ctx_obj: Any):

    ctx = _ctx_to_dict(ctx_obj)
    if not ctx or not ctx.get("sim_params"):
        st.info("컨텍스트에 시뮬레이션 정보가 없습니다. 좌측에서 고객을 선택하세요.")
        return

    st.markdown("##### 연금수령 시뮬레이션")
    if "지급기간_년" not in st.session_state:
        st.session_state["지급기간_년"] = 10  # <- 원하는 디폴트 

    # ★ 기본 디폴트 값 (요청값 반영)
    _def_평가기준일 = _date(2025, 9, 1)
    _def_생년월일   = _date(1968, 2, 15)
    _def_입사일     = _date(2009,10, 1)
    _def_퇴직일     = _date(2025, 9, 1)
    _def_IRP가입일   = _date(2014, 5, 1)
    _def_제도가입일  = _date(2014, 1, 1)
    _def_연금개시일  = _date(2025, 9, 1)  # 퇴직일 이후

    _def_과세제외_자기부담금   = 30_000_000
    _def_이연퇴직소득         = 500_000_000
    _def_세액공제자기부담금   = 150_000_000
    _def_운용손익             = 20_000_000
    _def_운용수익률           = 0.03

    with st.form("pension_inputs"):
        st.subheader("기본 정보(날짜)")
        d1, d2, d3 = st.columns(3)
        with d1:
                평가기준일 = st.date_input("평가기준일", value=_def_평가기준일)
                생년월일   = st.date_input("생년월일",   value=_def_생년월일)
                입사일     = date_input_optional("입사일", default=_def_입사일, key="입사일",
                                            help="퇴직소득이 없으면 '없음' 체크")

        with d2:
            퇴직일 = date_input_optional("퇴직일", default=_def_퇴직일, key="퇴직일",
                                    help="퇴직소득이 없으면 '없음' 체크")
            퇴직연금제도가입일 = st.date_input("퇴직연금 제도가입일", value=_def_제도가입일)
            IRP가입일 = date_input_optional("IRP 가입일", default=_def_IRP가입일, key="IRP가입일",
                                        help="미가입이면 '없음' 체크 → 평가기준일(당일 가입)로 대체")
            IRP가입일 = IRP가입일 if IRP가입일 is not None else 평가기준일

        # 🛠️ d1/d2에서 받은 값으로 '연금수령가능일' 즉시 산출
        _연금수령가능일_dt = pnsn_calculator.calc_연금수령가능일(
            생년월일=생년월일, IRP가입일=IRP가입일, 퇴직일=퇴직일
        )

        with d3:
            # 🛠️ 산출된 '연금수령가능일'을 보여주기(읽기전용)
            st.date_input("연금수령가능일 (자동 계산)", value=_연금수령가능일_dt, disabled=True,
                          help="퇴직일, 55세 되는 날, IRP 가입일 + 5년 중 가장 늦은 날")

            # 개시일은 사용자가 선택(기본값은 디폴트와 자동 산출값 중 더 늦은 날로 제안)
            _개시_디폴트 = _연금수령가능일_dt if _연금수령가능일_dt > _def_연금개시일 else _def_연금개시일
            연금개시일   = st.date_input("연금개시일(연금수령가능일 이후)", value=_개시_디폴트)
            운용수익률   = st.number_input("연 운용수익률(예: 0.03=3%)", value=_def_운용수익률, step=0.005, format="%.3f")
        
        b1, b2, b3 = st.columns(3)
        with b1:
            # 개시 나이(사용자 조정 가능)
            _auto_수령나이 = (연금개시일.year - 생년월일.year) - \
                             (1 if (연금개시일.month, 연금개시일.day) < (생년월일.month, 생년월일.day) else 0)
            st.caption("연금개시 연령: " f"{_auto_수령나이}세")
        with b2:
            # 근속년수(사용자 조정 가능)
            if 퇴직일 is not None and 입사일 is not None:
                근속월수 = (퇴직일.year - 입사일.year) * 12 + (퇴직일.month - 입사일.month)
                if 퇴직일.day < 입사일.day:
                    근속월수 -= 1
                _auto_근속년수 = math.ceil((근속월수 + 1) / 12)
            else:
                _auto_근속년수 = 0
            st.caption("근속년수: " f"{_auto_근속년수}년")
        with b3:
            _auto_연금수령연차 = max(0, 연금개시일.year - _연금수령가능일_dt.year) + 6 if 퇴직연금제도가입일 < _date(2013, 1, 1) else 1            
            st.caption("연금개시일 연금수령연차: " f"{_auto_연금수령연차}")
        

        submitted_main = st.form_submit_button("기본 정보 저장")

    st.subheader("연금소득 재원(원)")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        과세제외_자기부담금 = st.number_input("과세제외 자기부담금", value=_def_과세제외_자기부담금, step=100_000)
    with a2:
        이연퇴직소득 = st.number_input("이연퇴직소득(= IRP 입금 퇴직금)", value=_def_이연퇴직소득, step=1_000_000)
    with a3:
        세액공제자기부담금 = st.number_input("세액공제자기부담금", value=_def_세액공제자기부담금, step=100_000)
    with a4:
        운용손익 = st.number_input("운용손익", value=_def_운용손익, step=100_000)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_manual_tax_amount = st.checkbox("퇴직소득세액 직접입력")
    with c2:
        manual_tax_amount = st.number_input(
            "퇴직소득 산출세액(원)",
            value=0, step=1,
            disabled=not use_manual_tax_amount
        )
    if use_manual_tax_amount and 이연퇴직소득 > 0:
        st.caption(f"퇴직소득세율(입력 산출세액/이연퇴직소득): {manual_tax_amount/이연퇴직소득:.1%}")
    else:
        calc_퇴직소득세 = pnsn_calculator.calc_퇴직소득세(
            근속년수=_auto_근속년수, 이연퇴직소득=이연퇴직소득
        )
        st.caption(f"퇴직소득세율(계산기): {calc_퇴직소득세['퇴직소득세율']:.1%}")
        st.caption(f"퇴직소득세 산출세액(계산기): {calc_퇴직소득세['퇴직소득산출세액']:,} 원")


    st.caption(f"총평가금액(= 과세제외 자기부담금 + 이연퇴직소득 + 그외(=세액공제자기부담금 + 운용손익)): "
                f"{과세제외_자기부담금 + 이연퇴직소득 + 세액공제자기부담금 + 운용손익:,} 원")
    calc_퇴직소득세 = pnsn_calculator.calc_퇴직소득세(
        근속년수=_auto_근속년수, 이연퇴직소득=이연퇴직소득, 
    )

    st.subheader("지급 옵션")
    c1, c2, c3 = st.columns(3)    
    with c1:
        지급옵션 = st.selectbox("지급옵션", ["기간확정형", "금액확정형", "한도수령", "최소수령", "일시금"],
                                index=0, key="지급옵션")

    if 지급옵션 == "기간확정형":
        with c2:
            지급기간_년 = st.number_input(
                "지급기간_년(필수)", 
                min_value=1, 
                value=st.session_state.get("지급기간_년", 10), 
                step=1 
            )
        수령금액_년 = None

    elif 지급옵션 == "금액확정형":
        with c2:
            수령금액_년 = st.number_input(
                "수령금액_년(필수, 원)", 
                min_value=1, 
                value=12_000_000, 
                step=100_000
            )
        지급기간_년 = None

    else:
        # 한도수령, 최소수령일 경우
        지급기간_년, 수령금액_년 = None, None

    submitted_option = st.button("시뮬레이션 실행")

    if submitted_option:
        params = dict(
            평가기준일=평가기준일,
            # ↓ pnsn_calculator.simulate_pension이 '연금수령가능일'을 직접 쓰는 구조라면 이 값을 사용
            연금개시일=연금개시일,
            # (만약 내부에서 C25/C26로 재계산한다면 생년월일/퇴직일/IRP가입일을 넘기고 이 키는 빼세요)
            생년월일=생년월일,
            입사일=입사일,
            퇴직일=퇴직일,
            퇴직연금제도가입일=퇴직연금제도가입일,
            IRP가입일=IRP가입일,

            운용수익률=float(운용수익률),
            과세제외_자기부담금=int(과세제외_자기부담금),
            이연퇴직소득=int(이연퇴직소득),
            그외=int(세액공제자기부담금 + 운용손익),
     
            지급옵션=지급옵션,
            지급기간_년=int(지급기간_년) if 지급기간_년 else None,
            수령금액_년=int(수령금액_년) if 수령금액_년 else None,
        )
        if use_manual_tax_amount:
            params["퇴직소득산출세액_직접입력"] = int(manual_tax_amount)
        # 필수 검증
        if 지급옵션 == "기간확정형" and not params["지급기간_년"]:
            st.error("기간확정형에는 '지급기간_년'이 필요합니다."); st.stop()
        if 지급옵션 == "금액확정형" and not params["수령금액_년"]:
            st.error("금액확정형에는 '수령금액_년'이 필요합니다."); st.stop()

        try:
            with st.spinner("계산 중..."):
                df_capped = pnsn_calculator.simulate_pension(**params)
                # 일시금 지급옵션 추가 계산
                params_lump = params.copy()
                params_lump["지급옵션"] = "일시금"
                df_lump = pnsn_calculator.simulate_pension(**params_lump)                
                
            # 입력값 요약 + 결과 출력
            with st.container(border = True):
                st.markdown("##### 산출결과")
                m1, m2, m3, m4 = st.columns(4)
                _auto_현재나이 = (평가기준일.year - 생년월일.year) - \
                                (1 if (연금개시일.month, 연금개시일.day) < (생년월일.month, 생년월일.day) else 0)            
                with m1: st.metric("현재연령", f"{_auto_현재나이} 세")                    
                with m2: st.metric("연금개시일자", f"{연금개시일}")
                with m3: st.metric("연금개시연령", f"{_auto_수령나이}세")
                with m4: st.metric("연금개시금액", f"{int(df_capped[df_capped['지급회차']==1]['지급전잔액'].values[0]):,} 원")

                if {"총세액","실수령액","실제지급액"}.issubset(df_capped.columns):
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("총 연금수령액", f"{int(df_capped['실제지급액'].sum()):,} 원")                    
                    with m2: st.metric("총 세액 합계", f"{int(df_capped['총세액'].sum()):,} 원")
                    with m3: st.metric("실수령 합계", f"{int(df_capped['실수령액'].sum()):,} 원")
                    eff_tax_rate = df_capped['총세액'].sum() / df_capped['실제지급액'].sum() if df_capped['실제지급액'].sum() > 0 else 0
                    with m4: st.metric("실효세율", f"{eff_tax_rate:.1%}")
            
            with st.container(border = True):
                st.markdown("##### (일시금 수령 시)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("총 연금수령액", f"{int(df_lump['실제지급액'].sum()):,} 원")                    
                with m2: st.metric("총 세액 합계", f"{int(df_lump['총세액'].sum()):,} 원")
                with m3: st.metric("실수령 합계", f"{int(df_lump['실수령액'].sum()):,} 원")
                eff_tax_rate_lump = df_lump['총세액'].sum() / df_lump['실제지급액'].sum() if df_lump['실제지급액'].sum() > 0 else 0
                with m4: st.metric("실효세율", f"{eff_tax_rate_lump:.1%}")

            st.markdown("##### 산출결과 내역")
            # col_view = ["지급회차","나이","지급전잔액","한도","실제지급액","총세액","실수령액","세율","지급옵션"]
            # st.dataframe(
            #     style_dataframe(df_capped[col_view]),
            #     use_container_width=True, 
            #     hide_index=True,
            #     )
            # 1) 컬럼 생성
            df_capped["한도초과여부"] = df_capped.apply(
                lambda x: (
                    "한도 이내" if pd.isna(x["한도"]) or x["한도"] >= x["실제지급액"] 
                    else "한도 초과"
                ),
                axis=1
            )

            # 2) 스타일 적용 (DataFrame 먼저 자른 후 .style 사용)
            col_view = ["지급회차","나이","지급전잔액","한도","실제지급액",
                        "총세액","실수령액","세율","지급옵션","한도초과여부"]

            styled_df = style_dataframe(df_capped[col_view]).map(
                lambda v: "color:green;" if v=="한도 이내" else "color:red;",
                subset=["한도초과여부"]
            )

            # 3) 출력
            st.dataframe(styled_df, use_container_width=True, hide_index=True)       

            st.markdown("##### 산출결과 세부내역")
            st.dataframe(
                style_dataframe(df_capped),
                column_config={
                "연금지급일": st.column_config.DateColumn("연금지급일", format="YYYY-MM-DD"),
                "과세기간개시일": st.column_config.DateColumn("과세기간개시일", format="YYYY-MM-DD"),
                },
                use_container_width=True, 
                hide_index=True,
                )

            st.download_button(
                "CSV 다운로드",
                data=df_capped.to_csv(index=False).encode("utf-8-sig"),
                file_name="연금시뮬레이션_df_capped.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("시뮬레이션 중 오류가 발생했습니다.")
            st.exception(e)

