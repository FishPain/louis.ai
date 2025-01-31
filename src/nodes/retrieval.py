import re
from langchain.schema import HumanMessage
from src.templates import ResponseSufficency


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
    ### **Example Transforexcluded_file_idsmations**
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


def check_completeness_with_llm(state):
    model, query = state["model"], state["query"]
    context = "\n\n".join([f"- {doc.page_content}" for doc in state["retrieved_docs"]])

    """
    Uses the LLM to determine whether the retrieved legal context is sufficient 
    or if additional retrieval is necessary.
    """
    completeness_prompt = f"""
    You are a **highly skilled legal AI assistant** responsible for ensuring that retrieved legal documents
    contain all necessary information to answer the query **completely and accurately**.
    
    ---
    ### **User Query**
    {query}

    ### **Retrieved Legal Context**
    {context}
    ---

    ## **Task Instructions**  
    1. **Evaluate Completeness:**  
       - Determine whether the retrieved legal context contains **all necessary legal references, case laws, statutes, and key explanations** needed to answer the user’s query.  
       - If any referenced legal sections (e.g., "(2), (4)") or implicit legal concepts **are missing**, flag the response as incomplete.  

    2. **Identify Missing Information: (Focusing only on the Retrieved Legal Context)**  
       - If the retrieved context references **other legal sections** (e.g., "(3)", "(4)"), these sections must be **explicitly retrieved** before considering the response complete.  
       - If the retrieved documents **lack key legal principles**, suggest additional retrieval queries to obtain complete information.

    ---
    ## **Response Format**
    - **is_sufficient**: `true` if task instruction 1 and 2 are *strictly met*, otherwise `false`.  
    - **missing_queries**: A list of additional legal terms or key references that should be retrieved if `is_sufficient` is `false`.  
    """

    structured_output_parser = model.with_structured_output(ResponseSufficency)
    response = structured_output_parser.invoke([HumanMessage(content=completeness_prompt)])
    
    return response
