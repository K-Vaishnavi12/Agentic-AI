"""Legal Agent — MCA21-style registries, SEBI/regulatory changes, compliance.

ML enrichment: every legal item is run through the trained text classifier
which returns one of {compliance, regulatory, litigation, governance} along
with a confidence score. This complements the structured ``type`` field (which
may be missing in some sources).
"""
from typing import Any, Iterator
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent
from src.ml.registry import get_legal_classifier


def _ml_classify_legal(legal_data: list[dict]) -> tuple[list[dict], dict | None]:
    """Return (enriched_items, classifier_block) where classifier_block holds
    aggregate counts and the model name. Items get a ``ml_label`` field added.
    """
    clf = get_legal_classifier()
    enriched = list(legal_data) if legal_data else []
    if clf is None or not enriched:
        return enriched, None
    texts: list[str] = []
    idx_map: list[int] = []
    for i, it in enumerate(enriched):
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        body = (it.get("summary") or "").strip()
        text = (title + ". " + body).strip(". ").strip()
        if text:
            texts.append(text)
            idx_map.append(i)
    if not texts:
        return enriched, None
    try:
        preds = clf.predict(texts)
    except Exception:
        return enriched, None
    counts: dict[str, int] = {}
    for i, p in zip(idx_map, preds):
        if isinstance(enriched[i], dict):
            enriched[i] = {
                **enriched[i],
                "ml_label": p.label,
                "ml_confidence": round(p.confidence, 3),
            }
        counts[p.label] = counts.get(p.label, 0) + 1
    return enriched, {
        "model": clf.model_name,
        "n_classified": len(preds),
        "label_counts": counts,
    }


class LegalAgent(BaseAgent):
    agent_id = "legal"
    name = "Legal Agent"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        legal_data: list[Any] = context.get("legal_data", []) or []
        if not legal_data:
            return {
                "agent": self.agent_id,
                "summary": "No legal/regulatory data available. Run the legal pipeline to sync MCA21, SEBI, and compliance data.",
                "compliance_items": [],
                "regulatory_changes": [],
                "ml_classification": None,
            }

        enriched, ml_block = _ml_classify_legal(legal_data)
        req = context.get("user_requirement") or ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Legal Agent. Summarize MCA21/SEBI/compliance data briefly. List key deadlines and compliance items. Be concise."),
            ("human", "Data:\n{data}\n\n" + ("User: {requirement}\n\n" if req else "") + "Brief summary and compliance list."),
        ])
        chain = prompt | self.llm
        data_str = "\n".join(str(d) for d in enriched[:15])
        response = chain.invoke({"data": data_str, "requirement": req})

        # Use ML labels when present, fall back to the structured 'type' field
        def _is(item, label: str) -> bool:
            if not isinstance(item, dict):
                return False
            return (item.get("ml_label") == label) or (item.get("type") == label)

        result = {
            "agent": self.agent_id,
            "summary": response if isinstance(response, str) else getattr(response, "content", str(response)),
            "compliance_items": [d for d in enriched if _is(d, "compliance")][:15],
            "regulatory_changes": [d for d in enriched if _is(d, "regulatory")][:10],
            "litigation_items": [d for d in enriched if _is(d, "litigation")][:10],
            "governance_items": [d for d in enriched if _is(d, "governance")][:10],
            "ml_classification": ml_block,
        }
        self.remember(result["summary"], {"type": "legal_summary"})
        return result

    def stream_run(self, context: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield meta first, then stream LLM summary chunks."""
        legal_data: list[Any] = context.get("legal_data", []) or []
        if not legal_data:
            yield {"done": {
                "agent": self.agent_id,
                "summary": "No legal/regulatory data available. Run the legal pipeline to sync MCA21, SEBI, and compliance data.",
                "compliance_items": [], "regulatory_changes": [],
                "ml_classification": None,
            }}
            return
        enriched, ml_block = _ml_classify_legal(legal_data)

        def _is(item, label: str) -> bool:
            if not isinstance(item, dict):
                return False
            return (item.get("ml_label") == label) or (item.get("type") == label)

        compliance_items = [d for d in enriched if _is(d, "compliance")][:15]
        regulatory_changes = [d for d in enriched if _is(d, "regulatory")][:10]
        litigation_items = [d for d in enriched if _is(d, "litigation")][:10]
        governance_items = [d for d in enriched if _is(d, "governance")][:10]
        yield {"meta": {
            "agent": self.agent_id,
            "compliance_items": compliance_items,
            "regulatory_changes": regulatory_changes,
            "litigation_items": litigation_items,
            "governance_items": governance_items,
            "ml_classification": ml_block,
        }}
        req = context.get("user_requirement") or ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Legal Agent. Summarize MCA21/SEBI/compliance data briefly. List key deadlines and compliance items. Be concise."),
            ("human", "Data:\n{data}\n\n" + ("User: {requirement}\n\n" if req else "") + "Brief summary and compliance list."),
        ])
        chain = prompt | self.llm
        data_str = "\n".join(str(d) for d in enriched[:15])
        full_parts = []
        for chunk in chain.stream({"data": data_str, "requirement": req}):
            text = chunk.content if hasattr(chunk, "content") and chunk.content else ""
            if text:
                full_parts.append(text)
                yield {"text": text}
        summary = "".join(full_parts)
        self.remember(summary, {"type": "legal_summary"})
        yield {"done": {
            "agent": self.agent_id, "summary": summary,
            "compliance_items": compliance_items,
            "regulatory_changes": regulatory_changes,
            "litigation_items": litigation_items,
            "governance_items": governance_items,
            "ml_classification": ml_block,
        }}
