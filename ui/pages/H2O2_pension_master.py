import asyncio

import nest_asyncio
import streamlit as st
from agno.team import Team
from agno.tools.streamlit.components import check_password
from agno.utils.log import logger

from teams.pension_master import get_pension_master_team
from ui.css import CUSTOM_CSS
from ui.chat import render_chat_pane
from ui.panes import render_left_pane
from ui.panes.context_builder import render_context_inline
from ui.utils import (
    about_agno,
    add_message,
    display_tool_calls,
    example_inputs,
    initialize_team_session_state,
    selected_model,
)
from ui.state import PensionContext,SESSION_DEFAULTS
from ui.utils import inject_global_styles, ensure_session_defaults

nest_asyncio.apply()

st.set_page_config(
    page_title="한투 퇴직 연금 마스터",
    page_icon=":money_bag:",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
team_name = "pension_master_team"


async def header():
    # st.markdown("<h1 class='heading'>한투 퇴직 연금 마스터 | Team H2O2 </h1>", unsafe_allow_html=True)
    # st.markdown(
    #     "<p class='subheading'>맞춤형 데이터로 상담하고, 정책으로 판단하며, 시뮬레이션 실행까지 </p>",
    #     unsafe_allow_html=True,
    # )
    TITLE = "한투 <span class='accent'>퇴직연금 마스터</span> | Team H2O2"
    SUBTITLE = "맞춤 데이터로 상담하고, 정책으로 판단하며, 시뮬레이션으로 실행합니다"

    st.markdown("""
    <style>
    :root{
    /* Brand Colors (screen approximation) */
    --brand-blue:  #418FDE;  /* PANTONE 279C */
    --brand-brown: #603314;  /* PANTONE 732C */

    /* Neutrals */
    --title: #221915;
    --muted: #5A5A5A;
    --panel: #FFFFFF;

    /* Effects */
    --halo-1: 0 8px 28px rgba(65,143,222,0.18);
    --halo-2: 0 0 60px  rgba(65,143,222,0.16);
    --border: 1px solid rgba(96,51,20,0.16);

    /* Underline thickness */
    --u: 12px;
    }
    @media (prefers-color-scheme: dark){
    :root{
        --title:#F3F3F3;
        --muted:#CFCFCF;
        --panel:#111111;
        --border:1px solid rgba(255,255,255,0.10);
        --halo-1: 0 8px 28px rgba(65,143,222,0.30);
        --halo-2: 0 0 80px  rgba(65,143,222,0.28);
    }
    }

    /* Hero */
    .hero{
    position: relative;
    background: var(--panel);
    border: var(--border);
    border-radius: 18px;
    padding: 28px 28px 22px;
    margin: 8px 0 18px 0;
    box-shadow: var(--halo-1), var(--halo-2);
    overflow: hidden;

    /* 가운데 정렬 */
    text-align: center;
    }

    /* 상단에서 은은하게 비치는 블루 후광 */
    .hero::after{
    content:"";
    position:absolute; inset:-2px;
    pointer-events:none;
    border-radius: inherit;
    background:
        radial-gradient(1200px 360px at 20% -10%,
        rgba(65,143,222,0.18), transparent 60%);
    }

    /* Title */
    .hero h1{
    color: var(--title);
    font-size: clamp(28px, 4.2vw, 40px);
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin: 0 0 10px 0;
    text-shadow:
        0 2px 12px rgba(65,143,222,0.22),
        0 0 36px rgba(65,143,222,0.16);
    }

    /* === 강조 밑줄: 브라운→블루 그라디언트 === */
    .hero h1 .accent{
    /* 두 색이 함께 '밑줄'처럼 보이도록, 텍스트 아래에 gradient 배치 */
    background-image: linear-gradient(90deg, var(--brand-brown) 0%, var(--brand-blue) 100%);
    background-size: 100% var(--u);
    background-position: left calc(100% - 0px);
    background-repeat: no-repeat;

    /* 줄바꿈 시 각 줄에 동일한 밑줄이 적용되도록 */
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;

    padding: 0 3px; /* 밑줄 양끝 여백 */
    border-radius: 3px; /* 살짝 둥근 밑줄 느낌 */
    }

    /* Subtitle */
    .hero p{
    margin: 0;
    font-size: clamp(14px, 1.8vw, 16px);
    color: var(--muted);
    }

    /* 기능 칩이 있다면 중앙 정렬 유지 */
    .hero .chips{
    margin-top: 14px;
    display: flex; gap: 8px; flex-wrap: wrap;
    justify-content: center;
    }
    .hero .chip{
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(96,51,20,0.20);
    background: rgba(65,143,222,0.10);
    color: var(--title);
    }

    /* CTA 버튼(선택) */
    .hero .cta{
    display:inline-block; margin-top:16px;
    padding:10px 14px; border-radius:999px;
    text-decoration:none; color:#fff; font-weight:600;
    background: linear-gradient(90deg, var(--brand-brown), var(--brand-blue));
    border: 1px solid rgba(96,51,20,0.15);
    }
    .hero .cta:hover{ filter:brightness(0.96); }
    </style>
    """, unsafe_allow_html=True)

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

    team: Team
    if (
        team_name not in st.session_state
        or st.session_state[team_name]["team"] is None
        or st.session_state.get("selected_model") != model_provider.get('model_id')
    ):
        logger.info("---*--- Creating Team ---*---")
        team = get_pension_master_team(user_id=user_id, model_id=model_provider.get('model_id'))
        st.session_state["selected_model"] = model_provider.get('model_id')
    else:
        team = st.session_state[team_name]["team"]

    st.session_state["agent_cfg"] = {"provider": model_provider.get('provider'), "model": model_provider.get('model_id')}

    ####################################################################
    # Load Team Session from the database
    ####################################################################
    try:
        st.session_state[team_name]["session_id"] = team.load_session()
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
    await initialize_team_session_state(team_name)
    await header()
    await body()
    #await about_agno()


if __name__ == "__main__":
    if check_password():
        asyncio.run(main())
