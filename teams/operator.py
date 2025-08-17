from enum import Enum
from typing import List, Optional

from teams.finance_researcher import get_finance_researcher_team
from teams.multi_language import get_multi_language_team
from teams.pension_master import get_pension_master_team


class TeamType(Enum):
    FINANCE_RESEARCHER = "finance-researcher"
    MULTI_LANGUAGE = "multi-language"
    PENSION_MASTER = "pension-master"


def get_available_teams() -> List[str]:
    """Returns a list of all available team IDs."""
    return [team.value for team in TeamType]


def get_team(
    model_id: Optional[str] = None,
    team_id: Optional[TeamType] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
):
    if team_id == TeamType.FINANCE_RESEARCHER:
        return get_finance_researcher_team(
            model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode
        )
    elif team_id == TeamType.PENSION_MASTER:
        return get_pension_master_team(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    else:
        return get_multi_language_team(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
