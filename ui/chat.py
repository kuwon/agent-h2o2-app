# ui/chat.py
from __future__ import annotations
from typing import Any

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


# -----------------------------------------------------------------------------
# Session helpers
# -----------------------------------------------------------------------------
def _ensure_session_defaults() -> None:
    """context/messages 기본값 보장"""
    if "context" not in st.session_state or st.session_state["context"] is None:
        try:
            st.session_state["context"] = PensionContext()  # dataclass
        except Exception:
            st.session_state["context"] = {}
    if "messages" not in st.session_state or st.session_state["messages"] is None:
        st.session_state["messages"] = []


def _ctx_to_payload(ctx_obj: Any) -> dict:
    """PensionContext(dataclass)/dict/기타 → dict payload로 정규화"""
    if ctx_obj is None:
        return {}
    if is_dataclass(ctx_obj):
        try:
            return asdict(ctx_obj)
        except Exception:
            return getattr(ctx_obj, "__dict__", {}) or {}
    if isinstance(ctx_obj, dict):
        return ctx_obj
    return getattr(ctx_obj, "__dict__", {}) or {}


# -----------------------------------------------------------------------------
# Stream/event helpers
# -----------------------------------------------------------------------------
def _chunk_to_text(chunk: Any) -> str:
    """
    agno 스트리밍 청크를 UI 텍스트로 정규화.
    - str: 그대로
    - 객체: content / text / delta 속성 우선
    - dict: content / text / delta 키 우선
    - 그 외: 표시 안 함(툴 이벤트로 간주)
    """
    if isinstance(chunk, str):
        return chunk

    # 객체 속성
    for attr in ("content", "text", "delta"):
        try:
            val = getattr(chunk, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        except Exception:
            pass

    # dict 키
    if isinstance(chunk, dict):
        for key in ("content", "text", "delta"):
            v = chunk.get(key)
            if isinstance(v, str) and v.strip():
                return v

    return ""  # 표시 텍스트 없음 → 이벤트로 취급


def _format_tool_event(ev: Any) -> str:
    """
    툴 이벤트를 사람이 읽기 쉬운 마크다운으로 변환.
    다양한 이벤트 스키마에 안전하게 대응(duck-typing).
    """
    # dict 형태
    if isinstance(ev, dict):
        name = ev.get("tool_name") or ev.get("name") or "tool"
        args = ev.get("tool_args") or ev.get("args") or ev.get("parameters")
        out  = ev.get("output") or ev.get("result") or ev.get("content")
        md = f"**{name}**"
        if args:
            md += f"\n\n**Args**\n```json\n{args}\n```"
        if out:
            if isinstance(out, str):
                md += f"\n\n**Output**\n```\n{out}\n```"
            else:
                md += f"\n\n**Output**\n```json\n{out}\n```"
        return md

    # 객체 속성 추출
    name = getattr(ev, "tool_name", None) or getattr(ev, "name", None) or "tool"
    args = getattr(ev, "tool_args", None) or getattr(ev, "args", None) or getattr(ev, "parameters", None)
    out  = getattr(ev, "output", None) or getattr(ev, "result", None) or getattr(ev, "content", None)

    md = f"**{name}**"
    if args is not None:
        md += f"\n\n**Args**\n```json\n{args}\n```"
    if out is not None:
        if isinstance(out, str):
            md += f"\n\n**Output**\n```\n{out}\n```"
        else:
            md += f"\n\n**Output**\n```json\n{out}\n```"
    return md


# -----------------------------------------------------------------------------
# Main render
# -----------------------------------------------------------------------------
async def render_chat_pane(team: Team) -> None:
    """오른쪽 Chat 패널. 반드시 `await render_chat_pane(team)`로 호출하세요."""
    _ensure_session_defaults()
    ctx: PensionContext = st.session_state["context"]
    ctx_payload = _ctx_to_payload(ctx)  # ✅ team에게 넘길 컨텍스트

    st.divider()
    st.markdown("#### 채팅")

    # 1) 기존 대화 표시
    chat_holder = st.container()
    with chat_holder:
        for msg in st.session_state["messages"]:
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
    if not user_input:
        return

    # (1) 유저 메시지 저장/렌더
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with chat_holder:
        st.chat_message("user").markdown(user_input, unsafe_allow_html=True)

        # (2) 어시스턴트 스트리밍 출력
        resp_area = st.chat_message("assistant")
        placeholder = resp_area.empty()
        streamed = ""
        displayed_once_think = False

        # ✅ 한 줄짜리 컨텍스트 요약 캡션
        st.caption(f"ctx: customer={'yes' if ctx_payload.get('customer') else 'no'}, accounts={len(ctx_payload.get('accounts', []))}")

        try:
            run_response = run_pension_master(team, user_input, ctx_payload)

            tool_event_count = 0
            tool_events: list[Any] = []

            async for resp_chunk in run_response:
                piece = _chunk_to_text(resp_chunk)
                if not piece:
                    tool_event_count += 1
                    tool_events.append(resp_chunk)
                    continue

                streamed += piece
                visible, displayed_once_think = mask_thoughts(
                    streamed, displayed_once_think, final=False
                )
                placeholder.markdown(visible, unsafe_allow_html=True)

            final_visible, _ = mask_thoughts(streamed, displayed_once_think, final=True)

            maybe_coro = add_message(team.name, "assistant", final_visible)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

            # ✅ 요약 바(스트림 바이트/툴 이벤트)
            st.caption(f"stream bytes={len(streamed)}, tools={tool_event_count}")

            if tool_event_count > 0:
                with st.expander(f"🔧 내부 도구 사용 내역 ({tool_event_count})", expanded=False):
                    for ev in tool_events:
                        st.markdown(_format_tool_event(ev), unsafe_allow_html=True)
                # st.caption("도구 로그는 응답 본문과 분리해 표시합니다.")

        except Exception as e:
            st.error("대화 스트리밍 중 오류가 발생했습니다.")
            st.exception(e)
            return
