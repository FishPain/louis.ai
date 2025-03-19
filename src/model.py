from typing import List, TypedDict, Set
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage

from src.nodes.grader import (
    grade_compliance_node,
    grade_hallucination_node,
    grade_quality_node,
    intent_identification_node
)
from src.nodes.summarise import summarise_document_node
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
    complexity: str
    intent: str
    intent_type: str
    vectorstore_summary : str
    retrieved_docs: List[str]
    user_context: str
    system_context: str
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
    state["response"] = HumanMessage(
        content="The query appears to be out of scope for this system."
    )
    return state


def build_graph():
    """
    Build the workflow as a graph.
    1. what is the intent of this query
    2. does the intent includes wanting to analyse a document
    3. summarise the document if document is longer than x and add to user_context in query
    4. am I fit to answer this question
    5. do I need more system_context in answering this question
    6. which part of the intent am I missing that requires more system_context
    7. how can I search the vector store for the missing system_context
    8. let's put the information together and generate an answer
    9. is my system_context enough (go back to 3 if no)
    10. is there hallucination to my answer? (go back to 3 if no)
    11. is there biasness (go back to 2 if yes)
    12. is my answer relevant to my question (go back to 4 if no)
    13. does the intent includes wanting to generate a document
    """

    workflow = StateGraph(GraphState)

    workflow.add_node("intent_identification", intent_identification_node)
    workflow.add_node("summarise_document", summarise_document_node)
    workflow.add_node("complexity_ranking", complexity_scoring_node)
    workflow.add_node("retrieval_prompt", create_retrieval_prompt_node)
    workflow.add_node("response_constructor", response_constructor_node)
    workflow.add_node("vectorstore_recursive", recursive_vectorstore_node)
    # workflow.add_node("web_search", web_search_node)

    workflow.add_node("grader_hallucination", grade_hallucination_node)
    workflow.add_node("grader_quality", grade_quality_node)
    workflow.add_node("grader_compliance", grade_compliance_node)

    workflow.add_node("out_of_scope", handle_unrelated_content)

    # 1. what is the intent of this query
    workflow.add_edge(START, "intent_identification")
    # 2. does the intent includes wanting to analyse a document
    # 3. summarise the document if document is longer than x and add to user_context in query
    workflow.add_conditional_edges(
        "intent_identification",
        lambda state: state["intent"],
        {"summarise": "summarise_document", "qa": "complexity_ranking"},
    )
    workflow.add_edge("summarise_document", "complexity_ranking")
    # 4. am I fit to answer this question
    # 5. do I need more system_context in answering this question
    workflow.add_conditional_edges(
        "complexity_ranking",
        lambda state: state["complexity"],
        {
            Routing.COMPLEXITY_LOW: "response_constructor",
            Routing.COMPLEXITY_MEDIUM: "retrieval_prompt",
            Routing.COMPLEXITY_UNRELATED: "out_of_scope",
        },
    )
    # 8. is my system_context enough (go back to 6 if no)
    workflow.add_edge("retrieval_prompt", "vectorstore_recursive")
    # 9. let's put the information together and generate an answer
    workflow.add_edge("vectorstore_recursive", "response_constructor")
    # 10. is there hallucination to my answer? (go back to 6 if no)
    workflow.add_edge("response_constructor", "grader_hallucination")
    workflow.add_conditional_edges(
        "grader_hallucination",
        lambda state: state["hallucination"],
        {
            False: "grader_quality",
            True: "retrieval_prompt",
        },
    )
    # 11. is there biasness (go back to 2 if yes)
    workflow.add_conditional_edges(
        "grader_quality",
        lambda state: state["quality"],
        {
            True: "grader_compliance",
            False: "retrieval_prompt",
        },
    )
    # 12. is my answer relevant to my question (go back to 4 if no)
    workflow.add_conditional_edges(
        "grader_compliance",
        lambda state: state["compliance"],
        {
            True: END,
            False: "retrieval_prompt",
        },
    )

    # workflow.add_edge("web_search", END)
    workflow.add_edge("out_of_scope", END)

    return workflow
