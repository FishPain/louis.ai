from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

from src.nodes.retrieval import (
    check_completeness_with_llm,
    create_retrieval_prompt_node,
)


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
    excluded_file_ids = state.get(
        "excluded_file_ids", None
    )  # Default to None (include all files)

    # Step 1: Retrieve relevant documents
    retrieved_docs = vectorstore.similarity_search(
        query, top_k=len(excluded_file_ids) + 3, initial_k=len(excluded_file_ids) + 10
    )  # Retrieve more documents before filtering

    # extract only the content
    retrieved_docs = [doc[0] for doc in retrieved_docs]

    # Step 2: Filter out excluded documents
    if excluded_file_ids:
        retrieved_docs = [
            doc
            for doc in retrieved_docs
            if doc.metadata.get("id") not in excluded_file_ids
        ]

    # Step 3: Limit the result to top-k after filtering
    retrieved_docs = retrieved_docs[:3]  # Ensure we still return only 3 docs

    # Step 4: Check if retrieval was successful
    if not retrieved_docs:
        state["response"] = HumanMessage(
            content="No relevant legal documents were found in the database."
        )
        return state

    state["retrieved_docs"] = retrieved_docs
    state["context"] = "\n\n".join(
        [f"- {doc.page_content}" for doc in state["retrieved_docs"]]
    )
    return state


def response_constructor_node(state):
    model = state["model"]
    query = state["query"]
    user_context = state.get("user_context", None)
    system_context = state.get("system_context", None)
    intent = state["intent"]

    rag_prompt = f"""
You are a **highly experienced legal expert specializing in Singaporean law**, with deep expertise in **contract drafting, legislative interpretation, case law analysis, and delivering precise legal guidance**.

Your task is to provide a **clear, well-reasoned legal response** to the user's query, using the retrieved legal context and any user-provided information. Your answer must be **accurate**, **concise**, and **professionally structured**, suitable for a legal report or client communication.

---

### ✅ **Task Overview**

1. **Review the Provided Contexts**:
   - Carefully analyze the **Retrieved Legal Context** (from the knowledge base).
   - Consider the **User Uploaded Context** (such as contracts, documents, or user notes).

2. **Answer the User’s Query**:
   - Address the query **directly** based on the provided contexts.
   - Provide **legal reasoning**, referencing **specific statutes, clauses, or case law** whenever possible.
   - If the information provided is **incomplete**, explain this clearly and offer a **reasoned interpretation** based on existing knowledge and legal principles.  
   (⚠️ **Do not speculate or fabricate legal information**.)

3. **Maintain Professional Standards**:
   - Ensure the response is **legally accurate**, **logically sound**, and **clearly written**.
   - Avoid unnecessary jargon; be clear and precise.
   - Format the response in a **structured and professional** manner.

---
{
    f"### ✅ **Retrieved Legal Context (Knowledge Base)**:\n{system_context}\n---" if system_context else ""
}

{
    f"### ✅ **User Uploaded Context**:\n{user_context}\n---" if user_context else ""
}

### ✅ **User Query**:
{query}

---

### ✅ **User Intent**:
{intent}

---

### ✅ **Your Response Structure**:
Provide a professional legal opinion in the following structure:

**Legal Basis**:  
- Cite relevant laws, acts, regulations, or case precedents that apply.  
- E.g., "Under Section 14 of the Employment Act 1968 (Singapore)..."

**Analysis**:  
- Explain how the cited legal provisions apply to the user's specific situation.  
- Offer clear reasoning and interpretations where necessary.

**Conclusion**:  
- Summarize the legal position or recommended action.  
- Be concise and definitive where possible.

---

⚠️ **Important**:  
- If key information is missing, **state the limitations** clearly.  
- If additional documents or clarification are needed to provide a definitive answer, **recommend next steps**.

---

### ✅ **Begin your response below**:
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
        additional_state[
            "query"
        ] = f"""
        The retrieved documents are missing critical legal information regarding: **{missing_query}**.
        
        ### **Previous Query**
        {state["query"]}

        Please refine the search and retrieve documents that provide relevant information on **{missing_query}**.
        """
        additional_state = create_retrieval_prompt_node(
            additional_state
        )  # Generate a new retrieval prompt
        additional_result = vectorstore_node(
            additional_state
        )  # Retrieve missing information
        # Merge newly retrieved documents while avoiding duplicates
        state["retrieved_docs"].extend(additional_result["retrieved_docs"])

    return state
