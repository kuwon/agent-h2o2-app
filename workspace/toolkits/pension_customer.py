import json
import math
from typing import Callable, List

from agno.tools import Toolkit
from agno.tools.postgres import PostgresTools
from agno.utils.log import log_info, logger


class PensionCustomerTools(PostgresTools):
    def __init__(
        self,
        get_customer_summary: bool = True,
        enable_all: bool = False,
        
        **kwargs,
    ):
        # Initialize the toolkit with auto-registration enabled
        super().__init__(name="customer_db_tools", **kwargs)

        if get_customer_summary or enable_all:
            self.tools.append(self.get_customer_summary)

    def get_customer_summary(self, table: str, customer_id: int) -> str:
        """고객의 퇴직연금 요약(플랜 유형, 잔액, 납입률)을 반환합니다.
            Args:
                table: 조회할 테이블. 지정되지 않으면 kis_customers
                customer_id: 고객 ID
            Returns:
            사람이 읽을 수 있는 한국어 요약 문자열
        """
        with SessionLocal() as s:
            c = s.get(Customer, customer_id)
            if not c:
                return f"고객 {customer_id}를 찾을 수 없습니다."
            return (
                f"고객 {c.name}({c.id})님의 현재 플랜은 {c.plan_type}형이며, "
                f"적립금은 {int(c.balance):,}원, 납입률은 {c.contribution_rate}%입니다."
            )
        return json.dumps({"operation": "정보 요약", "result": result})