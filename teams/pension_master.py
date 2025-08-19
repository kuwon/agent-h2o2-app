from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama 
from agno.embedder.ollama import OllamaEmbedder
from agno.storage.postgres import PostgresStorage
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from agno.knowledge.url import UrlKnowledge
from agno.tools.knowledge import KnowledgeTools
from agno.vectordb.pgvector import PgVector, SearchType

from db.session import db_url
from teams.settings import team_settings

pension_docs = UrlKnowledge(
    urls=["https://github.com/kuwon/agent-h2o2-app/blob/main/resources/pension_rules.txt"],
    vector_db=PgVector(
        table_name="pension_knowledge",
        db_url=db_url,
        embedder=OllamaEmbedder()
    ),
)

knowledge_tools = KnowledgeTools(
    knowledge=pension_docs,
    think=True,
    search=True,
    analyze=True,
    add_few_shot=False,
)

simulation_agent = Agent(
    # TBD
    name="Simulation Agent",
    role="Analyze pension data",
    agent_id="simulation-agent",
    model=Ollama(
            id="qwen3:14b",
            host="host.docker.internal:11434"),
    # model=OpenAIChat(
    #     id=team_settings.gpt_4,
    #     max_completion_tokens=team_settings.default_max_completion_tokens,
    #     temperature=team_settings.default_temperature,
    # ),
    tools=[knowledge_tools],
    instructions=dedent(""" Show results only.
    """),
    storage=PostgresStorage(table_name="simulation-agent", db_url=db_url, auto_upgrade_schema=True),
    add_history_to_messages=True,
    num_history_responses=5,
    add_datetime_to_instructions=True,
    markdown=True,
)

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(
            id="qwen3:14b",
            host="host.docker.internal:11434"),
    tools=[DuckDuckGoTools(cache_results=True)],
    agent_id="web-agent",
    instructions=[
        "You are an experienced web researcher and news analyst!",
    ],
    show_tool_calls=True,
    markdown=True,
    storage=PostgresStorage(table_name="web_agent", db_url=db_url, auto_upgrade_schema=True),
)



def get_pension_master_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
):
    model_id = model_id or team_settings.qwen

    return Team(
        name="Pension Master Team",
        team_id="pension-master-team",
        mode="route",
        members=[simulation_agent],
        instructions=[
            "You are a team of pension simulation!. Use result of simulation_agent first.",
        ],
        session_id=session_id,
        user_id=user_id,
        description="You are a team of pension master!",
        model=Ollama(
            id="qwen3:14b",
            host="host.docker.internal:11434"),
        success_criteria="A good and simple answer",
        enable_agentic_context=True,
        expected_output="A good counseller for pension",
        storage=PostgresStorage(
            table_name="pension_master_team",
            db_url=db_url,
            mode="team",
            auto_upgrade_schema=True,
        ),
        debug_mode=debug_mode,
    )
