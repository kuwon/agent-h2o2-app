from textwrap import dedent
from typing import Optional

from agno.agent import Agent, AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama 
from agno.embedder.ollama import OllamaEmbedder
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.pgvector import PgVector, SearchType

from agents.settings import agent_settings
from db.session import db_url


def get_pension_policy(
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

    model_id = model_id or agent_settings.gpt_4_mini

    pension_agent = Agent(
        name="Pension Policy",
        agent_id="pension_policy",
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
        #tools=[DuckDuckGoTools()],
        tools=[],
        # Storage for the agent
        storage=PostgresAgentStorage(table_name="pension_sessions", db_url=db_url),
        # Knowledge base for the agent
        knowledge=AgentKnowledge(
            vector_db=PgVector(
                table_name="pension_knowledge"
                , db_url=db_url
                , embedder=OllamaEmbedder(id=agent_settings.open_embedding_model)
                ,search_type=SearchType.hybrid
            ) 
               
        ),
        # Description of the agent
        description=dedent(f"""\
            You have access to a knowledge base full of user-provided information and the capability to search the web if needed.
            Your responses should be clear, concise, and supported by citations from the knowledge base and/or the web.\
        """),
        # Instructions for the agent
        instructions=dedent("""\
            Respond to the user by following the steps below:

            1. Always search your knowledge base for relevant information
            - Rather than relying on your existing knowledge, first search the knowledge base for content similar to the question.
            - Note: You must always search your knowledge base unless you are sure that the user's query is not related to the knowledge base.

            2. Confirm user want to continue searching if no relevant information is found in your knowledge base               

            3. Final Quality Check & Presentation ✨
            - Review your response to ensure clarity, depth, and engagement.
            - Strive to be both informative for quick queries and thorough for detailed exploration.
            - Result Lnaguage should be Korean if query looks like Korean.

            4. In case of any uncertainties, clarify limitations and encourage follow-up queries.\
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
