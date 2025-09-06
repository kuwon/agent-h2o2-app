from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama 
from agno.embedder.ollama import OllamaEmbedder
from agno.storage.postgres import PostgresStorage
from agno.team.team import Team

from db.session import db_url
from teams.settings import team_settings

from agents.settings import agent_settings
from agents.intent_agent import get_intent
from agents.pension_account import get_pension_account
from agents.pension_policy import get_pension_policy

def get_pension_master_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
):
    model_id = model_id or team_settings.openai_economy

    return Team(
        name="Pension Master Team",
        team_id="pension-master-team",
        mode="route",
        members=[get_intent(), get_pension_policy(), get_pension_account()],
        instructions=dedent("""\
            Shared rules:
            - Evidence-only responses (Policy KB, PostgreSQL DB).
            - If evidence is missing/insufficient, respond with "unknown / verification needed."
            - Attach provenance for every figure/condition/definition (doc/article/date or SQL+params).
            - Protect PII; mask account numbers.
            - State "as-of" date if recency is uncertain.                            
        """),
        session_id=session_id,
        user_id=user_id,
        description=dedent("""\
                           The team answers retirement-pension queries only using the Policy KB and the account/transaction DB. 
                           No speculation or unsupported completions. 
                           If evidence is insufficient, explicitly say so and request further verification."""),
        #model=Ollama(id=agent_settings.qwen),
        model=OpenAIChat(id=model_id),
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
