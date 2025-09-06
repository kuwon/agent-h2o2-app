from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.postgres import PostgresTools

from agents.settings import agent_settings
from db.session import db_url
from db.settings import db_settings


def get_pension_account(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    model_id = model_id or agent_settings.openai_economy

    return Agent(
        name="Pension Account",
        agent_id="pension_account",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(
            id=model_id,
            max_completion_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        ),
        tools=[PostgresTools(**db_settings.get_db_info())],
        storage=PostgresAgentStorage(table_name="pension_account_sessions", db_url=db_url),
        description=dedent("""\
            고객/계좌 관련 DB 정보를 기반으로 질의에 응답한다.
        """),
        instructions=dedent("""\
            # 컨텍스트 활용
            - 호출 시 전달되는 context에 customer_id, account_id, accounts, dc_contracts 등이 포함될 수 있다.
            - 단순 조회/요약 요청이면 **context에 있는 데이터만으로 우선 답하라** (예: 개요 표, 합계/요약, 최근 n개 항목).
            - context에 정보가 부족할 때만 PostgresTools로 DB 질의를 수행하라.
            - 계좌번호 등 PII는 마스킹하라.

            # 응답 형식
            - 한국어로 간결하게 설명 + 필요한 경우 표/목록.
            - 수치에는 단위/기준일(as-of)을 함께 명시.
        """),
        markdown=True,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        show_tool_calls=False,
        debug_mode=debug_mode,
    )
