"""Startup Intelligence Agent — new businesses, founders, sectors, funding."""
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent


class StartupIntelligenceAgent(BaseAgent):
    agent_id = "startup"
    name = "Startup Intelligence Agent"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        startup_data = context.get("startup_data", [])
        if not startup_data:
            return {
                "agent": self.agent_id,
                "summary": "No startup data available. Run the startup pipeline to sync new registrations and funding data.",
                "new_entities": [],
                "sectors": [],
            }

        req = context.get("user_requirement") or ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Startup Intelligence Agent. Briefly summarize new businesses: founders, sectors, funding. List notable entities. Be concise."),
            ("human", "Data:\n{data}\n\n" + ("User: {requirement}\n\n" if req else "") + "Brief summary and list."),
        ])
        chain = prompt | self.llm
        data_str = "\n".join(str(d) for d in startup_data[:20])
        response = chain.invoke({"data": data_str, "requirement": req})

        result = {
            "agent": self.agent_id,
            "summary": response if isinstance(response, str) else getattr(response, "content", str(response)),
            "new_entities": [d for d in startup_data if isinstance(d, dict)][:20],
            "sectors": list(set(d.get("sector", "Other") for d in startup_data if isinstance(d, dict) and d.get("sector")))[:15],
        }
        self.remember(result["summary"], {"type": "startup_summary"})
        return result
