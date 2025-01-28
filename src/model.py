
from langchain.schema import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from src.constant import Routing
from src.templates import ComplexityRank
from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict

def complexity_ranking_node(state):
    """
    Uses an LLM to rank the complexity of the query based on what is already known in the vector store.
    Returns 'LOW', 'MEDIUM', or 'HIGH' complexity.
    """

    query = state["query"]
    model = state["model"]
    vectorstore_summary = state["vectorstore_summary"]

    # Define the decision prompt
    decision_prompt = f"""
    You are an intelligent assistant specializing in legal queries. Your job is to rank the complexity of the user's query 
    based on the knowledge already available in the vectorstore. 
    
    Here is the summary of the vectorstore contents:
    {vectorstore_summary}

    Based on this summary, rank the query's complexity as:
    - LOW: The query can be answered entirely using the vectorstore contents.
    - MEDIUM: The query is partially addressed by the vectorstore but may require additional reasoning from external resources.
    - HIGH: The query is outside the scope of the vectorstore contents and will require external resources.

    Query: {query}

    Complexity ({Routing.COMPLEXITY_LOW} / {Routing.COMPLEXITY_MEDIUM} / {Routing.COMPLEXITY_HIGH}):
    """

    # Set up the structured output parser
    structured_output_parser = model.with_structured_output(ComplexityRank)

    # Invoke the model with the prompt
    decision_response = structured_output_parser.invoke([HumanMessage(content=decision_prompt)])
    state["complexity"] = decision_response.complexity
    return state

def web_search_node(state):
    """
    Node: Handle external search using the Tavily API.
    """
    query = state["query"]
    model = state["model"]
    search = TavilySearchResults(max_results=2)
    tools = [search]
    query = state["query"]
    agent_executor = create_react_agent(model, tools)
    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    state["response"] = response
    return state

def vectorstore_node(state):
    query = state["query"]
    model = state["model"]
    retrieved_docs = state["db"].similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    rag_prompt = f"""
    You are a highly skilled legal expert specializing in Singaporean law, with extensive experience in drafting contracts, \
    interpreting legislation, and providing sound legal advice. Using the following retrieved legal context, provide a clear, \
    concise, and accurate response to the user's query. Where necessary, reference specific clauses or legal principles mentioned in the context.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    response = model.invoke([HumanMessage(content=rag_prompt)])
    state["response"] = response
    return state

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

def build_graph(query):
    """
    Build the workflow as a graph.
    """
    
    workflow = StateGraph(GraphState)

    workflow.add_node("complexity_ranking", complexity_ranking_node)
    workflow.add_node("vectorstore", vectorstore_node)
    workflow.add_node("web_search", web_search_node)


    workflow.add_edge(START, "complexity_ranking")
    workflow.add_conditional_edges(
        "complexity_ranking",
        lambda state: state["complexity"],
        {
            Routing.COMPLEXITY_LOW: "vectorstore",
            Routing.COMPLEXITY_MEDIUM: "vectorstore",
            Routing.COMPLEXITY_HIGH: "web_search",
        }
    )

    workflow.add_edge("vectorstore", END)
    workflow.add_edge("web_search", END)
    

    return workflow