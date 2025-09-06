from textwrap import dedent
from typing import Optional

from agno.agent import Agent, AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama 
from agno.embedder.ollama import OllamaEmbedder
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.postgres import PostgresTools
from agno.vectordb.pgvector import PgVector, SearchType

from agents.settings import agent_settings
from db.session import db_url
from db.settings import db_settings


def get_pension_account(
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

    model_id = model_id or agent_settings.openai_economy

    pension_agent = Agent(
        name="Pension Account",
        agent_id="pension_account",
        user_id=user_id,
        session_id=session_id,
        # model=Ollama(
        #     id=agent_settings.qwen
        #     , #host=agent_settings.local_ollama_host)
        # ),
        model=OpenAIChat(
           id=model_id,
           max_completion_tokens=agent_settings.default_max_completion_tokens,
           temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        ),
        # Tools available to the agent
        #tools=[DuckDuckGoTools()],
        tools=[PostgresTools(**db_settings.get_db_info())],
        # Storage for the agent
        storage=PostgresAgentStorage(table_name="pension_account_sessions", db_url=db_url),
        # Description of the agent
        description=dedent(f"""\
            고객, 계좌 관련된 테이블에서 사용자 질의에 필요한 정보들을 뽑아줌\
        """),
        # Instructions for the agent
        instructions=dedent("""\
            Respond to the user by following the steps below:
            1. 아래 정보를 가지고 DB에서 필요한 내용을 제공.
            - kis_customers: 고객 정보:  customer_id가 
            - kis_accounts: 계좌 정보: kis_customers와는 customer_id를 foreign_key로 엮여있음

            2. Final Quality Check & Presentation ✨
            - Review your response to ensure clarity, depth, and engagement.
            - Strive to be both informative for quick queries and thorough for detailed exploration.
            - Result Lnaguage should be Korean if query looks like Korean.

            3. In case of any uncertainties, clarify limitations and encourage follow-up queries.\
        """),
        additional_context=additional_context,
        # Format responses using markdown
        markdown=True,
        # Add the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Send the last 3 messages from the chat history
        add_history_to_messages=True,
        num_history_responses=3,
        # Add a tool to read the chat history if needed
        read_chat_history=True,
        # Show debug logs
        debug_mode=debug_mode,
    )
    return pension_agent
