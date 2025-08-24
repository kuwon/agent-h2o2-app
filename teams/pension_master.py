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

from agents.settings import agent_settings
from agents.intent import get_intent
from agents.pension_account import get_pension_account
from agents.pension_policy import get_pension_policy

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
        members=[get_intent(), get_pension_policy(), get_pension_account()],
        instructions=dedent("""\
            먼저 intent agent를 활용해서 사용자 쿼리를 분석하고 의도를 파악해.
            case A: 만약 사용자의 질문이 IRP/DC 등 퇴직연금 관련된 계좌 정보나 그 사용자에 대한 정보라고 한다면 pension_policy Agent를 사용해서 해당 정보를 획득해.
            case B: 퇴직연금 정책에 대한 질문을 한다면 pension_account agent가 가진 정책 knowledge를 활용하되, 이에 대한 고객/계좌 정보를 제대로 가지고 있는지를 확인해서 필요하면 그 정보를 추가로 얻도록 해야함.
            
            사용자 입력 언어를 잘 파악하고 해당 언어로 답변이 될수 있도록 유지를 해줘. 기본적으로는 한국어를 사용해.
            
            답변은 간결하게 해주고, 그에 대한 근거 및 원천에 대해서도 분명하게 표시해줘.
            
            결과를 모르면 모른다고 정확하게 이야기해줘.
                            
        """),
        session_id=session_id,
        user_id=user_id,
        description="퇴직연금에 대한 정보를 얻기 위해서 고객/계좌 정보나 관련 정책 정보를 조합하여 사용자가 원하는 정보를 얻게 도와주는 assistant!",
        model=Ollama(id=agent_settings.qwen),
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
