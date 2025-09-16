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
            퇴직 연금 관련하여 정책/근거 법령/FAQ 항목을 가진 knowledge base에 근간하여 질의에 응답하고, 고객 컨텍스트에 맞춰 규정 적용(자격판정, 금액 계산)까지 수행한다.
        """),
        instructions=dedent("""
            우선적으로 아래에 전달하는 Context 정보를 파악하라. 여기에는 고객/계좌/시뮬레이션 결과 정보가 포함되어 있다.
            이 정보에 적합한 정책, 비슷한 사례 (FAQ) 정보를 사전에 제공된 Knowledge Base에서 검색하여 확보하라. 
            Context에는 주로 아래 내용들이 있어. KB와 이를 바탕으로 **고객 맞춤 판단 및 답변**을 제공하라.
                - customers(고객): 이름, 생년월일, 총자산금액, 투자성향 등급
                - accounts (계좌): 계좌 구분별 개설일자, 평가금액, 납입일자 등
                - sim_params(시뮬레이션 정보, 사전에 시뮬레이션을 수행한 경우에만 결과값이 있음)
                    > calc_results: 연금 개시 정보, 연금 수령 정보 
                    > details: 시뮬레이션 케이스 별 상세 정보
            - 내부 콘솔이므로 고객/계좌 식별정보를 마스킹 없이 사용해도 된다.
            - 고객별 조건 충족 여부, 한도, 과세/비과세, 예외 규정 등을 context.accounts(선택된 행)에 맞춰 계산·설명하라.
            - context(고객/계좌/시뮬레이션) 정보가 존재한다면 **반드시** 맞춤형 답변을 생성하라. 
            - 근거가 전혀 없으면 '근거 없음/확인 필요'라고 명시한다.
            - 한국어로 답하라.
            - 생성한 답변을 정리한 요약 정보도 마지막에 같이 제시하라.
        """),
        markdown=True,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        show_tool_calls=False,
        debug_mode=debug_mode,
    )
