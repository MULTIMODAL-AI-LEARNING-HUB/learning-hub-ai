"""LangGraph workflow for multi-agent AI system."""

from typing import TypedDict
from src.agents.intent import classify_intent
from src.agents.retriever import retrieve
from src.agents.grader import grade_chunks
from src.agents.generator import generate_answer
from src.agents.reflection import reflect


class GraphState(TypedDict):
    query: str
    session_id: str
    user_id: str
    document_ids: list[str]
    intent: str
    sub_intent: str
    retrieved_chunks: list[dict]
    relevant_chunks: list[dict]
    current_answer: str
    citations: list[dict]
    needs_reflection: bool
    reflection_feedback: str
    final_answer: str


def intent_node(state: GraphState) -> GraphState:
    """Classify user intent."""
    result = classify_intent(state["query"])
    state["intent"] = result.get("intent", "qa")
    state["sub_intent"] = result.get("sub_intent", "default")
    return state


def retriever_node(state: GraphState) -> GraphState:
    """Retrieve relevant chunks from Qdrant."""
    doc_ids = state.get("document_ids") or None
    chunks = retrieve(state["query"], document_ids=doc_ids, limit=10)
    state["retrieved_chunks"] = chunks
    return state


def grader_node(state: GraphState) -> GraphState:
    """Grade relevance of retrieved chunks."""
    result = grade_chunks(state["query"], state["retrieved_chunks"])
    state["relevant_chunks"] = result.get("relevant_chunks", [])
    return state


def generator_node(state: GraphState) -> GraphState:
    """Generate answer based on relevant chunks."""
    result = generate_answer(state["query"], state["relevant_chunks"], intent=state["intent"])
    state["current_answer"] = result.get("answer", "")
    state["citations"] = result.get("citations", [])
    return state


def reflection_node(state: GraphState) -> GraphState:
    """Self-check answer quality."""
    result = reflect(state["current_answer"], state["relevant_chunks"], state["query"])
    state["needs_reflection"] = result.get("needs_reflection", False)
    state["reflection_feedback"] = result.get("feedback", "")
    return state


def finalize_node(state: GraphState) -> GraphState:
    """Finalize the answer."""
    state["final_answer"] = state["current_answer"]
    return state


def should_retry(state: GraphState) -> str:
    """Decide whether to retry generation."""
    if state.get("needs_reflection") and state.get("reflection_feedback"):
        return "retry"
    return "end"


def build_graph():
    """Build a simple workflow graph without LangGraph dependency."""
    def run_workflow(query: str, session_id: str, user_id: str, document_ids: list[str] | None = None) -> dict:
        state: GraphState = {
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "document_ids": document_ids or [],
            "intent": "",
            "sub_intent": "",
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "current_answer": "",
            "citations": [],
            "needs_reflection": False,
            "reflection_feedback": "",
            "final_answer": "",
        }

        # Step 1: Intent classification
        state = intent_node(state)

        # Route based on intent
        if state["intent"] in ("qa", "summarize"):
            # Step 2: Retrieve
            state = retriever_node(state)
            # Step 3: Grade
            state = grader_node(state)
            # Step 4: Generate
            state = generator_node(state)
            # Step 5: Reflect
            state = reflection_node(state)
            # Step 6: Retry if needed
            if should_retry(state) == "retry":
                state = generator_node(state)
            # Step 7: Finalize
            state = finalize_node(state)
        else:
            # For non-QA intents, just generate directly
            state = finalize_node(state)

        return {
            "answer": state["final_answer"] or state["current_answer"],
            "citations": state["citations"],
            "intent": state["intent"],
        }

    return run_workflow
