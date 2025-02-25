from typing import List, TypedDict, Set
from langgraph.graph import StateGraph, START, END
from src.nodes.grader import grade_compliance_node, grade_hallucination_node, grade_quality_node

class GraderGraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    query: str
    model: object
    response: List[object]
    hallucination: bool
    hallucination_reason: str
    quality: bool
    quality_reason: str
    compliance: bool
    compliance_reason: str

def build_response_grader_graph():
    """
    Builds graph for testing response grader
    """
    workflow = StateGraph(GraderGraphState)

    workflow.add_node("grader_hallucination", grade_hallucination_node)
    workflow.add_node("grader_quality", grade_quality_node)
    workflow.add_node("grader_compliance", grade_compliance_node)

    workflow.add_edge(START, "grader_hallucination")
    workflow.add_edge("grader_hallucination", "grader_quality")
    workflow.add_edge("grader_quality", "grader_compliance")
    workflow.add_edge("grader_compliance", END)

    return workflow

