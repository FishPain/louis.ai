from typing import List, TypedDict, Set
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage

from src.nodes.grader import (
    grade_compliance_node,
    grade_hallucination_node,
    grade_quality_node,
)
from src.nodes.scoring import complexity_scoring_node
from src.nodes.retrieval import create_retrieval_prompt_node
from src.nodes.search import (
    web_search_node,
    recursive_vectorstore_node,
    response_constructor_node,
)
from src.constant import Routing


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    query: str
    db: object
    model: object
    vectorstore_summary: str
    complexity: str
    retrieved_docs: List[str]
    response: List[object]
    depth: int
    excluded_file_ids: Set[str]
    hallucination: bool
    hallucination_reason: str
    quality: bool
    quality_reason: str
    compliance: bool
    compliance_reason: str


def handle_unrelated_content(state):
    state["response"] = HumanMessage(content="The query appears to be out of scope for this system.")
    return state


def build_graph():
    """
    Build the workflow as a graph.
    """

    workflow = StateGraph(GraphState)

    workflow.add_node("complexity_ranking", complexity_scoring_node)
    workflow.add_node("retrieval_prompt", create_retrieval_prompt_node)
    workflow.add_node("response_constructor", response_constructor_node)
    workflow.add_node("vectorstore_recursive", recursive_vectorstore_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_node("grader_hallucination", grade_hallucination_node)
    workflow.add_node("grader_quality", grade_quality_node)
    workflow.add_node("grader_compliance", grade_compliance_node)

    workflow.add_node("out_of_scope", handle_unrelated_content)

    workflow.add_edge(START, "complexity_ranking")

    workflow.add_conditional_edges(
        "complexity_ranking",
        lambda state: state["complexity"],
        {
            Routing.COMPLEXITY_LOW: "retrieval_prompt",
            Routing.COMPLEXITY_MEDIUM: "retrieval_prompt",
            Routing.COMPLEXITY_HIGH: "web_search",
            Routing.COMPLEXITY_UNRELATED: "out_of_scope",
        },
    )

    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("retrieval_prompt", "vectorstore_recursive")
    workflow.add_edge("vectorstore_recursive", "response_constructor")
    workflow.add_edge("response_constructor", "grader_hallucination")

    workflow.add_conditional_edges(
        "grader_hallucination",
        lambda state: state["hallucination"],
        {
            False: "grader_quality",
            True: "retrieval_prompt",
        },
    )
    workflow.add_conditional_edges(
        "grader_quality",
        lambda state: state["quality"],
        {
            True: "grader_compliance",
            False: "retrieval_prompt",
        },
    )
    workflow.add_conditional_edges(
        "grader_compliance",
        lambda state: state["compliance"],
        {
            True: END,
            False: "retrieval_prompt",
        },
    )
    workflow.add_edge("web_search", END)

    return workflow
