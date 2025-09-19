# 변경점 요약:
# - Enum 제거 → Literal 사용
# - Field(description=...) 모두 제거
# - response_model 그대로 유지

from textwrap import dedent
from typing import Optional, Literal
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage

from agents.settings import agent_settings
from db.session import db_url

from workspace.utils.db_key_eng_kor import KMAP_ACCOUNTS, KMAP_CUSTOMERS

# 👇 Enum 대신 Literal
class IntentResult(BaseModel):
    intent: Literal["account", "policy", "general"]
    customer_id: Optional[int] = None
    account_id: Optional[int]  = None
    topic: Optional[str]       = None


def get_intent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    model_id = model_id or agent_settings.openai_economy

    return Agent(
        name="Intent",
        agent_id="intent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(
            id=model_id,
            max_completion_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        ),
        tools=[],
        storage=PostgresAgentStorage(table_name="intent_sessions", db_url=db_url),
        description=dedent("""\
            대화의 의도를 분류해 계좌(agent:pension_account) 또는 정책(agent:pension_policy)로 라우팅하기 위한 전처리기.
        """),
        instructions=dedent(f"""\
            너는 퇴직연금 도메인의 라우팅 분류기다.
            1. 고객/계좌 관련해서는 다음 정보들이 있고 이 항목들을 바탕으로 고객 개인 정보/계좌/납입/전환/수익률/내역/계좌현황/고객군구분 등의 문의는 Account Agent로 라우팅
            - 고객 테이블 항목: {KMAP_CUSTOMERS}
            - 계좌 테이블 항목: {KMAP_ACCOUNTS}
            
            2. 정책 관련해서는 퇴직연금관련 정책/법령/정책/세제/제도, 규정, FAQ등의 정보가 있고, 이를 바탕으로 한 규정 적용/자격 판정/금액계산은 Agent Policy로 라우팅하라
            - 그 외 일반 설명/상담/스몰토크 -> general
            
            가능하면 customer_id, account_id, 정책 topic을 추출하라.
            반드시 JSON으로만 응답한다.
        """),
        response_model=IntentResult,   # ✅ 구조화 출력 유지
        markdown=False,
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        debug_mode=debug_mode,
    )
