import inspect

from textwrap import dedent
from typing import Optional, AsyncGenerator, Any, Dict

from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.storage.postgres import PostgresStorage

from db.session import db_url
from teams.settings import team_settings
from agents.settings import agent_settings

from agents.intent_agent import get_intent  # ✅ IntentLabel/IntentResult import 제거
from agents.pension_account import get_pension_account
from agents.pension_policy import get_pension_policy


def get_pension_master_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Team:
    """팀 객체 생성 (오케스트레이션은 run_pension_master에서 수행)."""
    model_id = model_id or team_settings.openai_economy

    return Team(
        name="pension_master_team",
        team_id="pension-master-team",
        mode="route",  # 내부 라우팅은 사용하지 않고, 아래 run_pension_master로 직접 오케스트레이션
        members=[get_intent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode),
                 get_pension_policy(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode),
                 get_pension_account(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)],
        instructions=dedent("""\
            Shared rules:
            - Use only the provided context, the Policy KB, and the PostgreSQL DB (via tools).
            - If evidence is missing/insufficient, respond with "unknown / verification needed."
            - Attach provenance for every figure/condition/definition (doc/article/date or SQL+params).
            - This is an internal console: it is OK to show personal identifiers (names, IDs, account numbers) from context/DB when it helps the answer.
            - Always prefer the given context (customer/accounts/sim_params); query DB/KB only to fill gaps.
        """),
        session_id=session_id,
        user_id=user_id,
        description=dedent("""\
            The team answers retirement-pension queries using the provided context first.
            If something is missing, account agent may query DB; policy agent cites KB.
            No speculation. If insufficient evidence, say so and ask for what’s missing.
        """),
        model=OpenAIChat(id=model_id),
        success_criteria="A good and simple answer",
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


# teams/pension_master.py (함수만 교체)
async def run_pension_master(team, message: str, context: dict):
    """
    안정화 + 가시 로그 버전:
    - intent 라벨/타겟 에이전트/컨텍스트 크기 로그
    - 스트리밍에서 텍스트 0바이트면 non-stream 폴백 강제
    - 항상 문자열만 yield (UI는 그대로 붙이기만)
    """
    import json, inspect

    def _to_dict(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        for fn in ("model_dump", "dict"):
            try:
                return getattr(obj, fn)()
            except Exception:
                pass
        return getattr(obj, "__dict__", {}) or {}

    def _chunk_to_text(ch) -> str:
        if isinstance(ch, str):
            return ch
        if isinstance(ch, dict):
            for key in ("content", "text", "delta"):
                v = ch.get(key)
                if isinstance(v, str) and v.strip():
                    return v
            return ""
        for attr in ("content", "text", "delta"):
            v = getattr(ch, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _build_runtime_context_prompt(ctx: dict) -> str:
        safe = {
            "customer": ctx.get("customer") or {},
            "customer_id": ctx.get("customer_id"),
            "accounts_selected": (ctx.get("accounts") or [])[:20],
            "sim_params": ctx.get("sim_params", {}),
        }
        guide = (
            "Use the runtime context below as the primary source of truth. "
            "If something is missing, query DB/KB via tools. "
            "Prefer Korean if user speaks Korean."
        )
        js = json.dumps(safe, ensure_ascii=False, indent=2)
        return f"{guide}\n<runtime_context>\n{js}\n</runtime_context>"

    async def _iterate_as_text(agent, msg: str, ctx: dict):
        """stream 결과를 무조건 str로 변환해 yield"""
        res = agent.arun(msg, stream=True, context=ctx)
        if inspect.isasyncgen(res):
            async for ev in res:
                txt = _chunk_to_text(ev)
                if txt:
                    yield txt
        else:
            out = await res
            yield _chunk_to_text(out)

    # ---------- 1) Intent ----------
    from agents.intent_agent import get_intent
    intent_agent = get_intent(
        model_id=team.model.id,
        user_id=team.user_id,
        session_id=team.session_id,
        debug_mode=False,
    )
    intent_res = await intent_agent.arun(message, stream=False, context=context)
    intent_obj = _to_dict(intent_res)
    label = str(intent_obj.get("intent", "general")).lower()

    # ---------- 2) Routing ----------
    if label == "account":
        from agents.pension_account import get_pension_account
        target = get_pension_account(model_id=team.model.id, user_id=team.user_id, session_id=team.session_id, debug_mode=False)
        chosen = "account"
    else:
        from agents.pension_policy import get_pension_policy
        target = get_pension_policy(model_id=team.model.id, user_id=team.user_id, session_id=team.session_id, debug_mode=False)
        chosen = "policy"

    # ---------- 3) Context injection ----------
    try:
        base_instr = getattr(target, "instructions", "") or ""
        runtime_ctx = _build_runtime_context_prompt(context or {})
        target.instructions = (base_instr + "\n\n" + runtime_ctx).strip()
    except Exception:
        pass

    # ---------- 4) Inline debug header (첫 줄로 한 번만) ----------
    debug_header = f"[route:{label}->{chosen}] ctx:customer={'yes' if context.get('customer') else 'no'}, accounts={len(context.get('accounts', []))}"
    yield f"<!-- {debug_header} -->\n"  # UI에는 보이지 않지만 소스에서 확인 가능

    # ---------- 5) Run (stream 우선, 0바이트면 non-stream 폴백) ----------
    got_text = False
    try:
        async for piece in _iterate_as_text(target, message, context):
            if isinstance(piece, str) and piece:
                got_text = True
                yield piece
    except Exception:
        got_text = False

    if not got_text:
        # non-stream 강제 폴백
        try:
            resp = await target.arun(message, stream=False, context=context)
            text = _chunk_to_text(resp)
            if isinstance(text, str) and text:
                yield text
            else:
                yield "응답을 생성하지 못했습니다. 질문을 다시 시도해 주세요."
        except Exception as e2:
            yield f"오류가 발생했습니다: {e2}"
