from os import getenv

from agno.playground import Playground

from agents.pension_policy import get_pension_policy
from teams.pension_master import get_pension_master_team
from workspace.dev_resources import dev_fastapi

######################################################
## Router for the Playground Interface
######################################################

pension_policy_agent = get_pension_policy(debug_mode=True)

# Create a playground instance
playground = Playground(agents=[pension_policy_agent], teams=[get_pension_master_team])

# Register the endpoint where playground routes are served with agno.com
if getenv("RUNTIME_ENV") == "dev":
    playground.serve(f"http://localhost:{dev_fastapi.host_port}")

playground_router = playground.get_async_router()
