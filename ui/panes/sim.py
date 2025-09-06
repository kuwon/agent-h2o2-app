
import streamlit as st
from ui.state import PensionContext
#from teams.pension_master import team_run
def render_sim_pane():
    st.subheader("시뮬레이션")
    ctx: PensionContext = st.session_state["context"]
    with st.form("sim_form"):
        amount = st.number_input("월 납입액 가정", min_value=0, value=int(ctx.sim_params.get("monthly", 300000)))
        years  = st.number_input("납입 기간(년)", min_value=0, value=int(ctx.sim_params.get("years", 10)))
        submitted = st.form_submit_button("시뮬레이션 실행")
        if submitted:
            ctx.sim_params={"monthly":int(amount),"years":int(years)}
            user_prompt="시뮬레이션을 실행해 주세요. 전제와 결과를 요약해 주세요."; team=st.session_state["team"]
            # import asyncio
            # async def _run():
            #     chunks=[]
            #     async for ch in team_run(team, user_prompt, ctx, stream=False): chunks.append(ch)
            #     return chunks
            # res=asyncio.run(_run()); st.success("실행 완료"); st.json(res[-1] if res else {})
    st.divider()
    if st.button("◀ 정보 보기로", use_container_width=True): st.session_state.left_view="info"; st.rerun()
