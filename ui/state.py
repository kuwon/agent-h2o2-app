
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from workspace.utils.model_providers import CHAT_MODELS

@dataclass
class PensionContext:
    customer_id: Optional[str] = None
    customer: List[Dict[str, Any]] = field(default_factory=list)
    customer_display: List[Dict[str, Any]] = field(default_factory=list)
    accounts: List[Dict[str, Any]] = field(default_factory=list)
    sim_params: Dict[str, Any] = field(default_factory=dict)
SESSION_DEFAULTS = {
    "left_view":"info",
    "messages":[],
    "context": PensionContext(), 
    "agent_cfg":{
        "provider":CHAT_MODELS.get('gpt-economy').get('provider'),
        "model":CHAT_MODELS.get('gpt-economy').get('model_id')
    }
}
