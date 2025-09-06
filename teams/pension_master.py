from textwrap import dedent
from typing import Optional, AsyncGenerator, Any

from agno.team.team import Team
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage

from db.session import db_url
from teams.settings import team_settings

from agents.settings import agent_settings
from agents.intent_agent import get_intent, IntentResult
from agents.pension_account import get_pension_account
from agents.pension_policy import get_pension_policy


def get_pension_master_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Team:
    model_id = model_id or team_settings.openai_economy

    intent_agent  = get_intent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    policy_agent  = get_pension_policy(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    account_agent = get_pension_account(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)

    return Team(
        name="Pension Master Team",
        team_id="pension-master-team",
        mode="route",
        members=[intent_agent, policy_agent, account_agent],
        instructions=dedent("""\
            Shared rules:
            - Evidence-only responses (Policy KB, PostgreSQL DB).
            - If evidence is missing/insufficient: reply "unknown / verification needed" and list what's missing.
            - Attach provenance for every figure/condition/definition (doc/article/date or SQL+params).
            - Protect PII; mask account numbers.
            - State "as-of" date if recency is uncertain.
        """),
        session_id=session_id,
        user_id=user_id,
        description=dedent("""\
            이 팀은 퇴직연금 문의에 대해 Intent → (Account/Policy)로 라우팅하고, Policy는 규정 적용까지 수행합니다.
        """),
        model=OpenAIChat(id=model_id),
        success_criteria="Accurate, sourced, and context-aware answers",
        enable_agentic_context=True,
        expected_output="A good counsellor for pension",
        storage=PostgresStorage(
            table_name="pension_master_team",
            db_url=db_url,
            mode="team",
            auto_upgrade_schema=True,
        ),
        debug_mode=debug_mode,
    )

def _extract_intent(intent_res: Any) -> str:
    """IntentResult/pydantic/dict/str 어떤 형태든 'account'|'policy'|'general'로 정규화."""
    # pydantic 모델
    if hasattr(intent_res, "model_dump"):
        data = intent_res.model_dump()
        return str(data.get("intent", "policy"))
    # dict
    if isinstance(intent_res, dict):
        return str(intent_res.get("intent", "policy"))
    # str
    if isinstance(intent_res, str):
        s = intent_res.strip().lower()
        if "account" in s: return "account"
        if "policy" in s:  return "policy"
        return "general"
    # fallback
    return "policy"

async def run_pension_master(
    team: Team,
    message: str,
    context: Any | None = None,
    stream: bool = True,
) -> AsyncGenerator:
    # 1) intent
    intent_agent = next(m for m in team.members if getattr(m, "agent_id", "") == "intent")
    intent_res = await intent_agent.arun(message, stream=False, context=context)
    intent = _extract_intent(intent_res)

    # 2) target agent 선택
    if intent == "account":
        target = next(m for m in team.members if getattr(m, "agent_id", "") == "pension_account")
    else:
        target = next(m for m in team.members if getattr(m, "agent_id", "") == "pension_policy")

    # 3) 실행 (context 포함)
    if stream:
        # ✅ 먼저 await 해서 스트림 객체(AsyncIterator)를 받습니다.
        stream_obj = await target.arun(message, stream=True, context=context)
        async for ch in stream_obj:
            yield ch
    else:
        res = await target.arun(message, stream=False, context=context)
        yield res
