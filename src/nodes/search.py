from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

from src.nodes.retrieval import check_completeness_with_llm, create_retrieval_prompt_node

def web_search_node(state):
    """
    Node: Handle external search using the Tavily API.
    """
    query = state["query"]
    model = state["model"]
    search = TavilySearchResults(max_results=3)
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
    vectorstore = state["db"]
    excluded_file_ids = state.get("excluded_file_ids", None)  # Default to None (include all files)

    # Step 1: Retrieve relevant documents
    retrieved_docs = vectorstore.similarity_search(query, k=len(excluded_file_ids)+3)  # Retrieve more documents before filtering

    # Step 2: Filter out excluded documents
    if excluded_file_ids:
        retrieved_docs = [
            doc for doc in retrieved_docs if doc.metadata.get("id") not in excluded_file_ids
        ]

    # Step 3: Limit the result to top-k after filtering
    retrieved_docs = retrieved_docs[:3]  # Ensure we still return only 3 docs

    # Step 4: Check if retrieval was successful
    if not retrieved_docs:
        state["response"] = "No relevant legal documents were found in the database."
        return state
    
    state["retrieved_docs"] = retrieved_docs
    return state


def response_constructor_node(state):
    model = state["model"]
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]

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


def recursive_vectorstore_node(state):
    """
    Recursive retrieval node: Performs search, checks completeness, and repeats if needed.
    """
    state = vectorstore_node(state)  # Perform initial retrieval
    state["depth"] += 1  # Increment depth counter
    for doc in state["retrieved_docs"]:
        state["excluded_file_ids"].add(doc.id)

    # Step 1: Check completeness
    completeness_result = check_completeness_with_llm(state)

    if completeness_result.is_sufficient or state["depth"] >= 3:
        # If retrieval is sufficient OR max depth is reached, return state
        return state

    # Step 2: Perform additional searches for missing references
    missing_queries = completeness_result.missing_queries

    if not missing_queries:
        return state  # No further queries needed

    for missing_query in missing_queries:
        additional_state = state.copy()
        additional_state["query"] = f"""
        The retrieved documents are missing critical legal information regarding: **{missing_query}**.
        
        ### **Previous Query**
        {state["query"]}

        Please refine the search and retrieve documents that provide relevant information on **{missing_query}**.
        """
        additional_state = create_retrieval_prompt_node(additional_state)  # Generate a new retrieval prompt
        additional_result = vectorstore_node(additional_state)  # Retrieve missing information
        # Merge newly retrieved documents while avoiding duplicates
        state["retrieved_docs"].extend(additional_result["retrieved_docs"])

    return state