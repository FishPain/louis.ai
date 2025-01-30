
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

    # Define the improved decision prompt
    decision_prompt = f"""
    You are an advanced AI system specializing in evaluating the complexity of legal queries. 

    Your task is to **analyze the user's query** in relation to the **knowledge stored in the vector store** and determine its complexity level.

    ### **Assessment Process**
    Follow these steps:
    1. **Understand the Query**  
       - What is the user asking?  
       - Does it involve a straightforward fact, a legal interpretation, or a multi-faceted legal issue?  
       
    2. **Analyze Available Knowledge**  
       - Review the provided summary of the vector store's contents.  
       - Identify if the vector store contains **direct answers, partial information, or lacks relevant content** for this query.  

    3. **Rank the Query Complexity**  
       Choose one of the following rankings based on the level of effort required to answer the query:  

       - **LOW**: The query is fully covered by the vector store. The answer can be directly retrieved.  
       - **MEDIUM**: Some relevant information is available, but additional **reasoning or minor external context** may be required.  
       - **HIGH**: The vector store lacks sufficient information, and **external resources or case law research** will be necessary.  

    ---
    ### **Vector Store Summary**
    {vectorstore_summary}

    ### **User Query**
    {query}

    ---
    ### **Final Decision**
    Rank the complexity as one of the following:  
    **({Routing.COMPLEXITY_LOW} / {Routing.COMPLEXITY_MEDIUM} / {Routing.COMPLEXITY_HIGH})**
    """

    # Set up the structured output parser
    structured_output_parser = model.with_structured_output(ComplexityRank)

    # Invoke the model with the prompt
    decision_response = structured_output_parser.invoke([HumanMessage(content=decision_prompt)])
    state["complexity"] = decision_response.complexity
    return state

def create_retrieval_prompt_node(state):
    """
    Node: Create an optimized retrieval prompt for similarity search in the vector store.
    """
    query = state["query"]
    model = state["model"]

    # Construct an optimized retrieval prompt
    prompt = f"""
    You are an expert in **optimizing queries for similarity-based vector retrieval**. Your goal is to **rewrite the given query** 
    into a **concise, search-optimized prompt** that improves the relevance of retrieved documents.

    ---
    ### **Optimization Guidelines**
    Follow these steps to generate the best possible retrieval prompt:

    1. **Extract Key Concepts**: Identify the most critical **legal terms, case names, and regulatory keywords** in the query.
    2. **Expand Query with Synonyms & Variations**: Reformulate the prompt using **different phrasings** that enhance similarity matching.
    3. **Remove Ambiguity**: Make the query **precise**, ensuring it aligns well with indexed documents.
    4. **Ensure Context Completeness**: Add essential **legal context** (e.g., jurisdiction, applicable laws, relevant case precedents).
    5. **Format for Search Optimization**: Structure the query so that it **maximizes cosine similarity scores in the embedding space**.

    ---
    ### **Example Transformations**
    **User Query**: "What are the rights of employees in Singapore regarding wrongful dismissal?"  
    **Optimized Retrieval Prompt**:  
    - "Singapore employment law: wrongful dismissal legal rights, case precedents, statutory protections."  
    - "Retrieve legal statutes, labor laws, and case law on wrongful termination in Singapore. Prioritize government regulations."  

    ---
    **Now, optimize the following query for vector retrieval.**  

    **User Query:**  
    {query}  

    **Optimized Retrieval Prompt:**  
    """

    response = model.invoke([HumanMessage(content=prompt)])
    state["response"] = response
    return state

def web_search_node(state):
    """
    Node: Handle external search using the Tavily API.
    """
    query = state["query"]
    model = state["model"]
    search = TavilySearchResults(max_results=1)
    tools = [search]
    query = state["query"]
    agent_executor = create_react_agent(model, tools)
    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    state["response"] = response
    return state

def vectorstore_node(state):
    """
    Node: Retrieves relevant legal documents from the vectorstore and generates a response.
    """
    query = state["query"]
    model = state["model"]
    vectorstore = state["db"]

    # Step 1: Retrieve relevant documents
    retrieved_docs = vectorstore.similarity_search(query, k=3)  # Increased k for more context

    # Step 2: Check if retrieval was successful
    if not retrieved_docs:
        state["response"] = "No relevant legal documents were found in the database."
        return state

    # Step 3: Extract and format retrieved context
    context = "\n\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    # Step 4: Construct an optimized RAG prompt
    rag_prompt = f"""
    You are a **highly skilled legal expert specializing in Singaporean law**, with extensive experience in **contract drafting, 
    legislative interpretation, case law analysis, and providing sound legal guidance**.

    ### **Task Instructions**
    1. **Analyze the Retrieved Legal Context**:
       - Review the extracted legal references, statutes, or precedents provided below.
       - Identify key points relevant to answering the query.
    
    2. **Provide a Well-Structured Legal Response**:
       - **Cite specific legal clauses, cases, or acts** when possible.
       - If the retrieved context is **incomplete**, provide a **reasoned legal interpretation** rather than making assumptions.
       - Ensure the response is **clear, concise, and legally accurate**.

    ---
    ### **Retrieved Legal Context**
    {context}
    ---

    ### **User Query**
    {query}

    ---
    ### **Your Answer (Format Your Response Properly)**
    - **Legal Basis**: [Reference specific clauses or laws]
    - **Explanation**: [Explain how the law applies]
    - **Conclusion**: [Summarize the legal standing]
    """

    # Step 5: Invoke the model
    response = model.invoke([HumanMessage(content=rag_prompt)])

    # Step 6: Store the response in the state
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

def build_graph():
    """
    Build the workflow as a graph.
    """
    
    workflow = StateGraph(GraphState)

    workflow.add_node("complexity_ranking", complexity_ranking_node)
    workflow.add_node("retrieval_prompt", create_retrieval_prompt_node)
    workflow.add_node("vectorstore", vectorstore_node)
    workflow.add_node("web_search", web_search_node)


    workflow.add_edge(START, "complexity_ranking")
    workflow.add_conditional_edges(
        "complexity_ranking",
        lambda state: state["complexity"],
        {
            Routing.COMPLEXITY_LOW: "retrieval_prompt",
            Routing.COMPLEXITY_MEDIUM: "retrieval_prompt",
            Routing.COMPLEXITY_HIGH: "web_search",
        }
    )

    workflow.add_edge("retrieval_prompt", "vectorstore")
    workflow.add_edge("vectorstore", END)
    workflow.add_edge("web_search", END)
    

    return workflow