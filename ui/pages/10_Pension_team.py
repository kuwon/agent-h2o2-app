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
    page_title="한투 퇴직 마스터",
    page_icon=":money_bag:",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
team_name = "pension_master_team"


async def header():
    st.markdown("<h1 class='heading'>한투 퇴직 마스터 Team</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subheading'>고객의 과거와 현재를 분석하여 미래를 설계하는 한투 퇴직 마스터</p>",
        unsafe_allow_html=True,
    )

async def body() -> None:
    ####################################################################
    # Initialize User and Session State
    ####################################################################
    user_id = "h2o2"

    ####################################################################
    # Model selector
    ####################################################################
    model_provider = await selected_model()

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
        st.subheader("챗봇 · 컨텍스트 · 실행")
        # 간단 컨텍스트 미리보기
        ctx: PensionContext = st.session_state["context"]
        st.json({
            "customer_id": ctx.customer_id,
            "accounts_preview": ctx.accounts[:2],
            "dc_contracts_preview": ctx.dc_contracts[:2],
            "sim_params": ctx.sim_params,
        })
        st.divider()
        render_chat_pane(team)


async def main():
    await initialize_team_session_state(team_name)
    await header()
    await body()
    await about_agno()


if __name__ == "__main__":
    if check_password():
        asyncio.run(main())
