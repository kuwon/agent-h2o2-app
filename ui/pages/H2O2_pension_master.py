import asyncio

import nest_asyncio
import streamlit as st
from agno.team import Team
from agno.tools.streamlit.components import check_password
from agno.utils.log import logger

from teams.pension_master import get_pension_master_team, run_pension_master
from ui.css import CUSTOM_CSS, TITLE_SUBTITLE_CSS
from ui.chat import render_chat_pane
from ui.panes import render_left_pane
from ui.panes.context_builder import render_context_inline
from ui.utils import (
    initialize_team_session_state,
    selected_model,
)
from ui.state import PensionContext,SESSION_DEFAULTS
from ui.utils import inject_global_styles, ensure_session_defaults

nest_asyncio.apply()

st.set_page_config(
    page_title="한투 퇴직 연금 마스터 챗봇",
    page_icon=":money_bag:",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
TEAM_KEY = "pension_master_team"


def _ensure_team_state():
    st.session_state.setdefault(TEAM_KEY, {})
    st.session_state[TEAM_KEY].setdefault("team", None)
    st.session_state[TEAM_KEY].setdefault("session_id", None)
    # 👇 세대 토큰(gen)과, 마지막으로 팀을 만든 시점의 토큰(built_gen)
    st.session_state[TEAM_KEY].setdefault("gen", 0)
    st.session_state[TEAM_KEY].setdefault("built_gen", -1)
    st.session_state.setdefault("selected_model", None)

def _rebuild_team(user_id: str, model_id: str):
    logger.info("---*--- Creating Team ---*---")
    team = get_pension_master_team(user_id=user_id, model_id=model_id)
    st.session_state[TEAM_KEY]["team"] = team
    st.session_state["selected_model"] = model_id

    # 세션 로드 (기존 로직 유지)
    try:
        st.session_state[TEAM_KEY]["session_id"] = team.load_session()
    except Exception:
        st.warning("Could not create Team session, is the database running?")
        # 실패 시 team을 None으로 되돌려도 됨
        st.session_state[TEAM_KEY]["team"] = None
        return None

    # 이번에 실제로 빌드한 gen을 기록
    st.session_state[TEAM_KEY]["built_gen"] = st.session_state[TEAM_KEY]["gen"]
    return team


async def header():
    # st.markdown("<h1 class='heading'>한투 퇴직 연금 마스터 | Team H2O2 </h1>", unsafe_allow_html=True)
    # st.markdown(
    #     "<p class='subheading'>맞춤형 데이터로 상담하고, 정책으로 판단하며, 시뮬레이션 실행까지 </p>",
    #     unsafe_allow_html=True,
    # )
    TITLE = "한투 <span class='accent'>퇴직연금 마스터</span> | Team H2O2"
    SUBTITLE = "맞춤 데이터로 상담하고, 정책으로 판단하며, 시뮬레이션으로 실행합니다"

    st.markdown(TITLE_SUBTITLE_CSS, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero">
    <h1>{TITLE}</h1>
    <p>{SUBTITLE}</p>
    <!-- 필요 시 기능 칩/버튼
    <div class="chips">
        <span class="chip">DB 고객/계좌 분석</span>
        <span class="chip">정책 KB 적용</span>
        <span class="chip">연금 시뮬레이션</span>
    </div>
    <a class="cta" href="#sim">시뮬레이션 시작하기</a>
    -->
    </div>
    """, unsafe_allow_html=True)  

async def body() -> None:
    ####################################################################
    # Initialize User and Session State
    ####################################################################
    user_id = "h2o2"

    ####################################################################
    # Model selector
    ####################################################################
    #model_provider =await selected_model()
    model_provider = {
        "model_id": "gpt-4o-mini",
        "provider": "openai"
    } 

    ####################################################################
    # Initialize Style
    ####################################################################

    inject_global_styles()
    ensure_session_defaults(SESSION_DEFAULTS)

    ####################################################################
    # Initialize Team
    ####################################################################

    _ensure_team_state()

    curr_model = model_provider.get("model_id")
    need_rebuild = (
        st.session_state[TEAM_KEY]["team"] is None
        or st.session_state.get("selected_model") != curr_model
        or st.session_state[TEAM_KEY]["built_gen"] != st.session_state[TEAM_KEY]["gen"]  # 👈 새로 추가
    )

    if need_rebuild:
        team = _rebuild_team(user_id=user_id, model_id=curr_model)
    else:
        team = st.session_state[TEAM_KEY]["team"]

    # 참고: agent config 기록(기존 유지)
    st.session_state["agent_cfg"] = {
        "provider": model_provider.get("provider"),
        "model": model_provider.get("model_id"),
    }

    ####################################################################
    # Load Team Session from the database
    ####################################################################
    try:
        st.session_state[TEAM_KEY]["session_id"] = team.load_session()
    except Exception:
        st.warning("Could not create Team session, is the database running?")
        return

    ####################################################################
    # Main
    ####################################################################
    left, gap, right = st.columns([0.48, 0.02, 0.50], gap="small")
    with left:
        render_left_pane(st.session_state.get("left_view", "info"))

    with gap:
        st.markdown('<div class="v-sep"></div>', unsafe_allow_html=True)


    with right:
        # 탭 폰트/패딩 키우기
        st.markdown("""
        <style>
        /* 탭 버튼 자체 크기 키우기 */
        button[role="tab"] {
            font-size: 18px !important;   /* 글자 크게 */
            font-weight: 600 !important;  /* 굵게 */
            padding: 14px 22px !important;/* 패딩 키움 */
            line-height: 1.4 !important;  /* 줄간격 여유 */
        }

        /* 선택된 탭 강조 */
        button[aria-selected="true"][role="tab"] {
            background-color: #f0f2f6 !important;
            border-bottom: 3px solid #2b6cb0 !important; /* 파란 밑줄 */
            font-weight: 700 !important;
        }
        </style>
        """, unsafe_allow_html=True)


        st.subheader("대화 · 시뮬레이션 · 컨텍스트")

        tab_chat, tab_sim, tab_ctx = st.tabs(["💬 AI에게 물어보기", "📈 시뮬레이션 계산기", "🧩 Context 미리보기/편집"])

        with tab_ctx:
            render_context_inline(expanded=False)

        with tab_sim:
            try:
                from ui.panes.sim import render_sim_pane
            except Exception:
                st.info("시뮬레이션 Pane 모듈이 아직 없습니다. ui/panes/sim.py를 추가하세요.")
            else:
                ctx_obj = st.session_state.get("context")
                render_sim_pane(ctx_obj)   # 현재 컨텍스트 기반

        with tab_chat:
            await render_chat_pane(team)



async def main():
    await initialize_team_session_state(TEAM_KEY)
    await header()
    await body()
    #await about_agno()


if __name__ == "__main__":
    if check_password():
        asyncio.run(main())
