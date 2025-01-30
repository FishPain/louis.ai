from langchain.schema import HumanMessage

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
