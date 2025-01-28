from langgraph.graph import StateGraph
from langchain_core.schema import BaseMessage


def decide_action_node(state):
    """
    Node: Decide complexity of the query.
    """
    query = state["query"]
    vectorstore_summary = state["vectorstore_summary"]

    complexity = decide_action(query, chat_model, vectorstore_summary)
    state["complexity"] = complexity
    return state


def vectorstore_node(state):
    """
    Node: Handle RAG (vectorstore) retrieval.
    """
    context = "\n".join([doc.page_content for doc in state["retrieved_docs"]])
    query = state["query"]
    
    response = f"""
    Using the vectorstore content:
    {context}

    Query: {query}
    Answer:
    """
    state["response"] = response
    return state


def web_search_node(state):
    """
    Node: Handle web search.
    """
    query = state["query"]
    response = search_tool.invoke({"query": query})  # Example tool usage
    state["response"] = response
    return state


# Define the workflow as a graph
workflow = StateGraph(initial_state={"query": "Your legal question", "retrieved_docs": [], "vectorstore_summary": "Summary"})

workflow.add_node("decide_action", decide_action_node)
workflow.add_node("vectorstore", vectorstore_node)
workflow.add_node("web_search", web_search_node)

# Add conditional edges
workflow.add_conditional_edges(
    "decide_action",
    lambda state: state["complexity"],  # Routing condition
    {
        "LOW": "vectorstore",
        "MEDIUM": "vectorstore",  # Example, MEDIUM could still use vectorstore
        "HIGH": "web_search",
    }
)

# Compile and run
app = workflow.compile()
final_state = app.run()
print(final_state["response"])