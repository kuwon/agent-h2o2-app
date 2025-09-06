import asyncio 
import nest_asyncio

import streamlit as st
from streamlit import components

from ui.state import PensionContext
from ui.utils import (
    mask_thoughts,
    add_message,
)

from agno.team import Team

#from teams.pension_team import team_run

async def render_chat_pane(team: Team):
    nest_asyncio.apply()
    ctx: PensionContext = st.session_state["context"]
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
            resp_area = st.chat_message("assistant")
            placeholder = resp_area.empty()
            streamed = ""
            displayed_once_think = False

            run_response = await team.arun(user_input, stream=True)
            async for resp_chunk in run_response:
            #for chunk in run_agent_stream(user_input, ctx, debug=debug_on):
                streamed += resp_chunk
                visible, displayed_once_think = mask_thoughts(streamed, displayed_once_think)
                placeholder.markdown(visible, unsafe_allow_html=True)

            # (3) 최종 텍스트 저장
            final_visible, _ = mask_thoughts(streamed, displayed_once_think)
            await add_message(team.name, "assistant", final_visible)
            #st.session_state.messages.append({"role": "assistant", "content": final_visible})

