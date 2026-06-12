"""LLM evaluation harness — measures the orchestrator briefing on:
  • factual coverage  — does the briefing mention every agent's key fields?
  • ROUGE-L           — overlap with a reference (sub-agent summaries concat'd)
  • latency           — wall-clock time to produce the briefing
  • length            — token-ish proxy via word count

Runs once against the existing pipeline data, no extra LLM calls per metric.
"""
from __future__ import annotations
import time
from typing import Any

from src.ml.utils.metrics import rouge_l


# Keywords each agent is expected to surface in the briefing
_EXPECTED_KEYWORDS = {
    "stock": ["stock", "market", "price"],
    "legal": ["legal", "compliance", "regulator"],
    "startup": ["startup", "sector", "founder"],
    "risk": ["risk", "danger", "volatil"],
    "shareholder": ["shareholder", "dividend", "equity"],
}


def _coverage(briefing: str, agent_outputs: dict[str, Any]) -> dict[str, float]:
    """For each agent, fraction of expected keywords present in the briefing."""
    text = (briefing or "").lower()
    out: dict[str, float] = {}
    for name, keywords in _EXPECTED_KEYWORDS.items():
        if name not in agent_outputs:
            continue
        hits = sum(1 for k in keywords if k in text)
        out[name] = hits / max(1, len(keywords))
    return out


def _word_count(text: str) -> int:
    return len((text or "").split())


def evaluate_briefing(verbose: bool = True) -> dict:
    """Run the orchestrator once, score the briefing. Skips gracefully if
    Ollama is unreachable.
    """
    try:
        from src.pipelines import get_context_from_pipelines
        from src.orchestrator import run_executive_briefing
    except Exception as e:
        return {"error": f"orchestrator import failed: {e}"}

    try:
        ctx = get_context_from_pipelines()
    except Exception as e:
        return {"error": f"pipeline failed: {e}"}

    t0 = time.time()
    try:
        result = run_executive_briefing(ctx, stream_stdout=False)
    except Exception as e:
        return {"error": f"LLM unreachable: {e}",
                "hint": "Start Ollama (`ollama serve`) and pull the model."}
    latency = time.time() - t0

    briefing = result.get("executive_briefing") or ""
    agent_outputs = result.get("agent_outputs") or {}

    # Reference = concatenated sub-agent summaries (ROUGE-L approximates
    # how much of the upstream signal the briefing preserves).
    reference = "\n".join(
        (a.get("summary") or "") for a in agent_outputs.values() if isinstance(a, dict)
    )

    metrics = {
        "latency_seconds": round(latency, 3),
        "briefing_words": _word_count(briefing),
        "reference_words": _word_count(reference),
        "rouge_l_vs_reference": round(rouge_l(reference, briefing), 4),
        "coverage": _coverage(briefing, agent_outputs),
        "n_agents": len(agent_outputs),
    }
    metrics["coverage_macro"] = round(
        sum(metrics["coverage"].values()) / max(1, len(metrics["coverage"])), 4
    )
    if verbose:
        print(f"[llm-eval] latency={metrics['latency_seconds']}s "
              f"rouge-L={metrics['rouge_l_vs_reference']} "
              f"cov={metrics['coverage_macro']}")
    return metrics
