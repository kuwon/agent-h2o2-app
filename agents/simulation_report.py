from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.utils.log import logger

from agents.settings import agent_settings
from db.session import db_url
from db.settings import db_settings


def get_simulation_report(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    model_id = model_id or agent_settings.openai_economy
    logger.info("get_simulation_report")

    return Agent(
        name="Simulation Report",
        agent_id="simulation_report",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(
            id=model_id,
            max_completion_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        ),
        tools=[],
        storage=PostgresAgentStorage(table_name="simulation_report_sessions", db_url=db_url),
        description=dedent("""\
            당신은 퇴직연금 시뮬레이션 결과를 고객/상담사에게 설명하는 전문 리포트 작성자입니다.
            - 제공된 시뮬레이션 입력 (dict_sumul_input), 결과 (dict_sumul_output), 그리고 결과 상세 내역(df_capped)만 근거로 작성하세요.
            - 숫자는 3자리 콤마, 화폐 단위 '원' 사용. 세율/수익률은 % 표기.
            - 과도한 확신 금지. 가정/한계를 명시.
            - 한국어로 간결하지만 신뢰감 있는 문체.
        """),
        instructions=dedent("""
            리포트 형식(Markdown):
            # 연금 시뮬레이션 요약
            - 주요 입력 정보
            - 연금 개시 관련 주요 정보 (개시일, 기준금액 등)
            - 연금 수령 관련 주요 정보 (실수령액, 세금 등)

            ## 고객 상황에 따른 해설
            - ○○ 조건에서는 △△ 옵션이 상대적으로 유리할 수 있음 (이유). 지급 옵션별로 비교한 결과도 활용
            - 세무/현금흐름 유의사항.

            ## 상세 내역 요약
            - 데이터 기준/시점
            - 주요 파라미터
            - 지급액/실수령액/한도초과 여부 등이 주요하게 달라지는 회차/수령연차/지급일자 포인트 분석 내역 
        """),
        markdown=True,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        show_tool_calls=False,
        debug_mode=debug_mode,
    )
