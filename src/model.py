from typing import List, TypedDict, Set
from langgraph.graph import StateGraph, START, END

from src.nodes.scoring import complexity_scoring_node
from src.nodes.retrieval import create_retrieval_prompt_node
from src.nodes.search import web_search_node, recursive_vectorstore_node, response_constructor_node
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

    workflow.add_edge(START, "complexity_ranking")

    workflow.add_conditional_edges(
        "complexity_ranking",
        lambda state: state["complexity"],
        {
            Routing.COMPLEXITY_LOW: "retrieval_prompt",
            Routing.COMPLEXITY_MEDIUM: "retrieval_prompt",
            Routing.COMPLEXITY_HIGH: "web_search",
        },
    )

    workflow.add_edge("retrieval_prompt", "vectorstore_recursive")
    workflow.add_edge("vectorstore_recursive", "response_constructor")
    workflow.add_edge("response_constructor", END)
    workflow.add_edge("web_search", END)

    return workflow
