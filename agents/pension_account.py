from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.postgres import PostgresTools

from agents.settings import agent_settings
from db.session import db_url
from db.settings import db_settings


def get_pension_account(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    model_id = model_id or agent_settings.openai_economy

    return Agent(
        name="Pension Account",
        agent_id="pension_account",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(
            id=model_id,
            max_completion_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature if model_id != "o3-mini" else None,
        ),
        tools=[PostgresTools(**db_settings.get_db_info())],
        storage=PostgresAgentStorage(table_name="pension_account_sessions", db_url=db_url),
        description=dedent("""\
            고객/계좌 관련 DB 정보를 기반으로 질의에 응답한다.
        """),
        instructions=dedent("""
            너의 목적은 고객/계좌 관련 질문에 대해 **먼저 context를 사용**해 정확히 답하는 것이다.
            - context.customer, context.accounts(선택된 행들), context.sim_params 를 1차 근거로 활용하라.
            - 부족하거나 확인이 필요한 데이터만 PostgresTools로 DB에서 조회해 보완하라.
            - 이 환경은 내부 콘솔이므로 고객 이름, 고객번호, 계좌번호 등 식별정보를 **마스킹 없이 그대로** 보여도 된다.
            - 답변에는 사용한 근거(컨텍스트/SQL 컬럼·조건 등)를 간단히 밝혀라.
            - 한국어 질문에는 한국어로 답하라.

            사용 가능 데이터(예시 컬럼):
            - kis_customers(customer_id, customer_name, brth_dt, tot_asst_amt, cust_ivst_icln_grad_cd)
            - kis_accounts(account_id, customer_id, acnt_type, prd_type_cd, acnt_bgn_dt, expd_dt, etco_dt, rtmt_dt, midl_excc_dt,
                        acnt_evlu_amt, copt_year_pymt_amt, other_txtn_ecls_amt, rtmt_incm_amt, icdd_amt, user_almt_amt,
                        sbsr_almt_amt, utlz_erng_amt, dfr_rtmt_taxa)

            절차:
            1) 질문에서 필요한 항목을 식별하고, 먼저 context에서 값을 찾는다.
            2) context로 충분하지 않다면 **최소 컬럼만** DB에서 질의한다(과한 SELECT 지양).
            3) 결과가 없거나 불확실하면 '데이터 없음/추가 정보 필요'라고 명시한다.
        """),
        markdown=True,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        show_tool_calls=False,
        debug_mode=debug_mode,
    )
