from textwrap import dedent
from typing import Optional

from agno.agent import Agent, AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.vectordb.pgvector import PgVector
from agno.embedder.openai import OpenAIEmbedder

from agents.settings import agent_settings
from db.session import db_url


def get_pension_policy(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    model_id = model_id or agent_settings.openai_economy

    return Agent(
        name="Pension Policy",
        agent_id="pension_policy",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(
            id=model_id,
            max_completion_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        ),
        tools=[],  # applicator는 본 에이전트 내부 절차로 수행
        storage=PostgresAgentStorage(table_name="pension_policy_sessions", db_url=db_url),
        knowledge=AgentKnowledge(
            vector_db=PgVector(
                table_name="kis_pension_knowledge_openai",
                db_url=db_url,
                embedder=OpenAIEmbedder(),
            )
        ),
        description=dedent("""\
            정책/FAQ/법령 질의에 응답하고, 고객 컨텍스트에 맞춰 규정 적용(자격판정, 금액 계산)까지 수행한다.
        """),
        instructions=dedent("""\
            ## 1) RAG 우선
            - 먼저 지식베이스(PgVector)에서 관련 조항/FAQ/문서를 검색한다.
            - 근거 인용(출처/문서/조항/시행일)을 반드시 포함한다.
            - 근거가 없거나 불충분하면 "모름/검증 필요"라고 답하고, 어떤 정보가 더 필요한지 명시한다.

            ## 2) 컨텍스트 기반 맞춤 적용 (applicator 내장)
            - 호출 시 전달되는 context에 customer_id, employment, accounts, dc_contracts 등이 포함된다.
            - 사용자의 질문이 "적용/자격/금액" 판단이면:
              1) 관련 규칙을 요점 JSON으로 구조화 (조건/예외/계산/효력기간).
              2) context의 필드와 매핑해 충족/불충족 근거를 조목조목 제시.
              3) 부족한 필드는 추가 질문으로 요청.
              4) 계산식이 있으면 단계별로 계산(상한/하한/예외 포함) → 최종 결과.

            ## 3) 표현
            - 한국어로 간결하게, 표/불릿으로 핵심 먼저.
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
