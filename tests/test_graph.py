import pytest
from unittest.mock import patch

from src.workflows.graph import build_graph, finalize_node, should_retry

@pytest.mark.asyncio
async def test_should_retry_logic():
    state_no_retry = {"needs_reflection": False, "reflection_feedback": ""}
    assert should_retry(state_no_retry) == "end"

    state_retry = {"needs_reflection": True, "reflection_feedback": "Clarify answer"}
    assert should_retry(state_retry) == "retry"

@pytest.mark.asyncio
async def test_finalize_node():
    state = {"current_answer": "Test Answer", "final_answer": ""}
    res = finalize_node(state)
    assert res["final_answer"] == "Test Answer"

@pytest.mark.asyncio
async def test_graph_execution_fallback():
    with patch("src.workflows.graph.classify_intent", return_value={"intent": "qa", "sub_intent": "default"}):
        with patch("src.workflows.graph.retrieve_for_course", return_value=[]):
            with patch("src.workflows.graph.grade_chunks", return_value={"relevant_chunks": []}):
                with patch("src.workflows.graph.generate_answer", return_value={"answer": "Mocked AI Response", "citations": []}):
                    with patch("src.workflows.graph.reflect", return_value={"needs_reflection": False, "feedback": ""}):
                        graph_runner = build_graph()
                        result = await graph_runner(
                            query="What is AI?",
                            session_id="session-1",
                            user_id="user-1",
                            course_id="course-1"
                        )

                        assert result["answer"] == "Mocked AI Response"
                        assert result["intent"] == "qa"
