"""Shareholder Agent — dividends, equity movements, board decisions."""
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent


class ShareholderAgent(BaseAgent):
    agent_id = "shareholder"
    name = "Shareholder Agent"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        shareholder_data = context.get("shareholder_data", [])
        if not shareholder_data:
            return {
                "agent": self.agent_id,
                "summary": "No shareholder data available. Sync dividends, equity movements, and board decisions.",
                "dividends": [],
                "equity_movements": [],
                "board_decisions": [],
            }

        req = context.get("user_requirement") or ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Shareholder Agent. Briefly summarize dividends, equity moves, board decisions. Be concise."),
            ("human", "Data:\n{data}\n\n" + ("User: {requirement}\n\n" if req else "") + "Brief summary."),
        ])
        chain = prompt | self.llm
        data_str = "\n".join(str(d) for d in shareholder_data[:15])
        response = chain.invoke({"data": data_str, "requirement": req})

        result = {
            "agent": self.agent_id,
            "summary": response if isinstance(response, str) else getattr(response, "content", str(response)),
            "dividends": [d for d in shareholder_data if isinstance(d, dict) and d.get("type") == "dividend"][:10],
            "equity_movements": [d for d in shareholder_data if isinstance(d, dict) and d.get("type") == "equity"][:10],
            "board_decisions": [d for d in shareholder_data if isinstance(d, dict) and d.get("type") == "board"][:10],
        }
        self.remember(result["summary"], {"type": "shareholder_summary"})
        return result
