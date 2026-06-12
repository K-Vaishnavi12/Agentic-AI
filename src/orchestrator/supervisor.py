"""LangGraph Supervisor — Chairman's Master Orchestrator."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from src.llm import get_llm
from src.agents import (
    StockAgent,
    LegalAgent,
    StartupIntelligenceAgent,
    RiskAgent,
    ShareholderAgent,
)


class AgentState(TypedDict):
    """State for the supervisor graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: dict[str, Any]
    agent_outputs: dict[str, Any]
    next_agent: str | None
    briefing: str | None
    stream_stdout: bool


# Sub-agents
AGENTS = {
    "stock": StockAgent(),
    "legal": LegalAgent(),
    "startup": StartupIntelligenceAgent(),
    "risk": RiskAgent(),
    "shareholder": ShareholderAgent(),
}


def _invoke_agent(agent_name: str, context: dict[str, Any]) -> dict[str, Any]:
    agent = AGENTS.get(agent_name)
    if not agent:
        return {"error": f"Unknown agent: {agent_name}"}
    return agent.run(context)


def run_all_agents(state: AgentState) -> dict:
    """Run all sub-agents in parallel and collect outputs."""
    context = state["context"]
    agent_outputs = {}
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as executor:
        futures = {executor.submit(_invoke_agent, name, context): name for name in AGENTS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                agent_outputs[name] = future.result()
            except Exception as e:
                agent_outputs[name] = {"error": str(e), "summary": f"Error: {e}"}
    # Keep fixed order for consistent briefing sections
    agent_outputs = {name: agent_outputs[name] for name in AGENTS}
    return {"agent_outputs": agent_outputs, "next_agent": None}


def supervisor_node(state: AgentState) -> dict:
    """Supervisor decides we run all agents then synthesize (single pass)."""
    return {"next_agent": "all"}


def synthesize_briefing(state: AgentState) -> dict:
    """Chairman's executive briefing from all agent outputs. Streams to stdout when stream_stdout is True."""
    import sys
    outputs = state.get("agent_outputs") or {}
    llm = get_llm(temperature=0.3)
    summaries = []
    for agent_name, data in outputs.items():
        s = data.get("summary", str(data))
        summaries.append(f"[{agent_name}]\n{s}")
    combined = "\n\n".join(summaries)

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Chairman's Master Orchestrator. Synthesize the following sub-agent reports into ONE unified executive briefing. Structure: (1) Executive summary (2) Market & stock (3) Legal & compliance (4) Startup/competitive (5) Risk (6) Shareholder. Be concise; one paragraph per section. No external references."),
        ("human", "Sub-agent reports:\n{reports}\n\nProduce the single executive briefing."),
    ])
    chain = prompt | llm
    stream_stdout = state.get("stream_stdout")

    if stream_stdout:
        briefing_parts = []
        for chunk in chain.stream({"reports": combined}):
            if hasattr(chunk, "content") and chunk.content:
                text = chunk.content
                briefing_parts.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()
        briefing = "".join(briefing_parts)
    else:
        briefing = chain.invoke({"reports": combined})
        if hasattr(briefing, "content"):
            briefing = briefing.content
    return {"briefing": briefing, "messages": [HumanMessage(content=briefing)]}


def create_supervisor_graph() -> StateGraph:
    """Build the LangGraph: supervisor -> run_all_agents -> synthesize_briefing -> END."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("run_agents", run_all_agents)
    workflow.add_node("synthesize", synthesize_briefing)

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "run_agents")
    workflow.add_edge("run_agents", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow


def run_executive_briefing(context: dict[str, Any] | None = None, stream_stdout: bool = True) -> dict[str, Any]:
    """Run the full graph and return executive briefing + agent outputs. When stream_stdout is True, the final briefing is streamed to stdout as it is generated."""
    graph = create_supervisor_graph().compile()
    initial: AgentState = {
        "messages": [],
        "context": context or {},
        "agent_outputs": {},
        "next_agent": None,
        "briefing": None,
        "stream_stdout": stream_stdout,
    }
    result = graph.invoke(initial)
    return {
        "executive_briefing": result.get("briefing"),
        "agent_outputs": result.get("agent_outputs", {}),
        "messages": result.get("messages", []),
    }
