# ui/chat.py
import asyncio
import nest_asyncio
from dataclasses import asdict, is_dataclass

import streamlit as st
from streamlit import components
from agno.team import Team

from ui.state import PensionContext
from ui.utils import (
    mask_thoughts,
    add_message,   # async/sync 모두 대응
)

from teams.pension_master import run_pension_master

nest_asyncio.apply()


def _ensure_session_defaults():
    # context/messages 기본값 보장
    if "context" not in st.session_state or st.session_state["context"] is None:
        try:
            st.session_state["context"] = PensionContext()  # dataclass
        except Exception:
            st.session_state["context"] = {}
    if "messages" not in st.session_state or st.session_state["messages"] is None:
        st.session_state["messages"] = []


def _ctx_to_payload(ctx_obj):
    """PensionContext(dataclass)/dict/기타 → dict payload로 정규화"""
    if ctx_obj is None:
        return {}
    if is_dataclass(ctx_obj):
        try:
            return asdict(ctx_obj)
        except Exception:
            # dataclass지만 asdict 실패 시 __dict__ 폴백
            return getattr(ctx_obj, "__dict__", {}) or {}
    if isinstance(ctx_obj, dict):
        return ctx_obj
    # 기타 객체는 __dict__ 사용
    return getattr(ctx_obj, "__dict__", {}) or {}


def _chunk_text(chunk) -> str:
    """스트리밍 청크를 문자열로 안전 변환."""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if content is not None:
        return str(content)
    if isinstance(chunk, dict):
        c = chunk.get("content")
        if c is not None:
            return str(c)
    return str(chunk)


async def render_chat_pane(team: Team):
    """오른쪽 Chat 패널. 반드시 `await render_chat_pane(team)`로 호출하세요."""
    _ensure_session_defaults()
    ctx: PensionContext = st.session_state["context"]
    ctx_payload = _ctx_to_payload(ctx)  # ✅ team에게 넘길 컨텍스트

    st.divider()
    st.markdown("#### 채팅")

    # 1) 기존 대화 표시
    chat_holder = st.container()
    with chat_holder:
        for msg in st.session_state.messages:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            st.chat_message(role).markdown(msg.get("content", ""), unsafe_allow_html=True)

    # 2) 입력창 앵커 (렌더 직후 스크롤)
    st.markdown("<div id='right-chat-input-anchor'></div>", unsafe_allow_html=True)

    # 3) 입력창
    user_input = st.chat_input("질문을 입력하세요. (예: 현재 컨텍스트 기반으로 DC 관련 규정 설명)")

    # 4) 렌더 직후 앵커로 스크롤
    components.v1.html(
        """
        <script>
        const el = document.getElementById('right-chat-input-anchor');
        if (el) el.scrollIntoView({behavior: 'auto', block: 'end'});
        </script>
        """,
        height=0,
    )

    # 5) 전송/응답 스트리밍
    if user_input:
        # (1) 유저 메시지 저장/렌더
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_holder:
            st.chat_message("user").markdown(user_input, unsafe_allow_html=True)

            # (2) 어시스턴트 스트리밍 출력
            resp_area = st.chat_message("assistant")
            placeholder = resp_area.empty()
            streamed = ""
            displayed_once_think = False

            try:
                # ✅ ctx를 team에 전달
                run_response = run_pension_master(team, user_input, context=ctx_payload, stream=True)
                async for resp_chunk in run_response:
                    piece = _chunk_text(resp_chunk)
                    streamed += piece
                    visible, displayed_once_think = mask_thoughts(streamed, displayed_once_think)
                    placeholder.markdown(visible, unsafe_allow_html=True)
            except Exception as e:
                st.error("대화 스트리밍 중 오류가 발생했습니다.")
                st.exception(e)
                return

            # (3) 최종 텍스트 저장
            final_visible, _ = mask_thoughts(streamed, displayed_once_think)

            agent_name = getattr(team, "name", None) or getattr(team, "team_id", None) or "global"

            try:
                await add_message(agent_name, "assistant", final_visible)  # async 버전
            except TypeError:
                # add_message가 sync 함수인 경우
                add_message(agent_name, "assistant", final_visible)
            except Exception as e:
                st.warning("메시지 저장(add_message) 중 문제가 발생했지만 화면 출력은 완료되었습니다.")
                st.exception(e)

