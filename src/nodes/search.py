from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage


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
    model = state["model"]
    vectorstore = state["db"]

    # Step 1: Retrieve relevant documents
    retrieved_docs = vectorstore.similarity_search(
        query, k=3
    )  # Increased k for more context

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
