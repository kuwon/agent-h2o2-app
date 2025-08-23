import json
import math
from typing import Callable, List

from agno.tools import Toolkit
from agno.utils.log import log_info, logger


class PensionPolicyTools(Toolkit):
    def __init__(
        self,
        pension_eligibility: bool = True,
        enable_all: bool = False,
        
        **kwargs,
    ):
        # Build the include_tools list based on enabled functions
        tools: List[Callable] = []
        if pension_eligibility or enable_all:
            tools.append(self.pension_eligibility)

        # Initialize the toolkit with auto-registration enabled
        super().__init__(name="pension-calculator", tools=tools, **kwargs)

    def pension_eligibility(self, acnt_evlu_amt: float, rcve_odyr: float) -> str:
        """caculate pension_eligibility and return the result.

        Args:
            acnt_evlu_amt (float): account_evaluation_amount
            rcve_odyr (float): receive_order_of_year

        Returns:
            str: JSON string of the result.
        """
        result = acnt_evlu_amt / (11-rcve_odyr) * 120
        log_info(f"with acnt_evlu_amt: {acnt_evlu_amt} and rcve_odyr {rcve_odyr} to get {result}")
        return json.dumps({"operation": "addition", "result": result})