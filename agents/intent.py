from textwrap import dedent
from typing import Optional

from enum import Enum
from pydantic import BaseModel, Field

from agno.agent import Agent, AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama 
from agno.embedder.ollama import OllamaEmbedder
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.vectordb.pgvector import PgVector, SearchType

from agents.settings import agent_settings
from db.session import db_url


class IntentLabel(str, Enum):
    account = "account" # 고객 개인 계좌/납입/전환 등
    policy = "policy" # 법령/정책/FAQ
    general = "general" # 일반 설명/소프트 질문


class IntentResult(BaseModel):
    intent: IntentLabel = Field(..., description="user 질문의 1차 의도")
    customer_id: Optional[int] = Field(None, description="고객 관련일 때 customer_id")
    customer_id: Optional[int] = Field(None, description="계좌 관련일 때 account_id")
    topic: Optional[str] = Field(None, description="정책/FAQ 주요 토픽 키워드")

def get_intent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    additional_context = ""
    if user_id:
        additional_context += "<context>"
        additional_context += f"You are interacting with the user: {user_id}"
        additional_context += "</context>"

    model_id = model_id or agent_settings.qwen

    intent_agent = Agent(
        name="Intent",
        agent_id="intent",
        user_id=user_id,
        session_id=session_id,
        model=Ollama(
            id=agent_settings.qwen
            , #host=agent_settings.local_ollama_host)
        ),
        #model=OpenAIChat(
        #    id=model_id,
        #    max_completion_tokens=agent_settings.default_max_completion_tokens,
        #    temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        #),
        # Tools available to the agent
        tools=[],
        # Storage for the agent
        storage=PostgresAgentStorage(table_name="intent_sessions", db_url=db_url),
        
        # Description of the agent
        description=dedent(f"""\
            대화의 의도를 분류. 기본적으로는 퇴직연금에 대한 질의 응답을 하고, 계좌관련이라던지 정책 관련된 부분들에 대해서 제대로 응답하기 위하여 사용자의 질문에서 의도를 분류\
        """),
        # Instructions for the agent
        instructions=dedent("""\
            너는 퇴직연금 도메인의 라우팅 분류기다.
            - 고객 개인 정보/계좌/납입/전환/수익률 문의 -> account
            - 법령/정책/세제/제도/FAQ -> policy
            - 그 외 일반 설명/상담/스몰토크 -> general
            가능하면 customer_id, accountid, 정책 topic을 추출해라.
            숫자가 아닌 고객번호는 None으로 둔다.
            이후에 활용하기 편하도록 json 형태로 결과를 생성해야한다.\
        """),
        response_model=IntentResult,
        # Format responses using markdown
        markdown=True,
        # Add the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Send the last 3 messages from the chat history
        #add_history_to_messages=True,
        #num_history_responses=3,
        # Add a tool to read the chat history if needed
        #read_chat_history=True,
        # Show debug logs
        debug_mode=debug_mode,
    )
    return intent_agent
