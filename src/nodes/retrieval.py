import re
from langchain.schema import HumanMessage
from src.templates import ResponseSufficency


def create_retrieval_prompt_node(state):
    """
    Node: Create an optimized retrieval prompt for similarity search in the vector store.
    """

    query = state["query"]
    model = state["model"]
    hallucination = state.get("hallucination", False)
    hallucination_reason = state.get("hallucination_reason", "")
    quality = state.get("quality", False)
    quality_reason = state.get("quality_reason", "")
    compliance = state.get("compliance", False)
    compliance_reason = state.get("compliance_reason", "")

    # Construct an optimized retrieval prompt
    prompt = f"""
	You are an expert in **optimizing legal queries for similarity-based vector retrieval**. Your task is to **rewrite the user query** into a **concise, search-optimized prompt** to improve the relevance and accuracy of retrieved documents from a legal vector database.

	---

	### ✅ **Guidelines for Optimizing the Retrieval Prompt**
	1. **Extract Key Legal Concepts**: Focus on critical legal terms, case names, jurisdictions, and regulations present in the query.
	2. **Expand with Synonyms and Variations**: Include alternative phrasings or terminology to enhance vector similarity matching.
	3. **Clarify and Disambiguate**: Remove vague language; ensure precision and legal accuracy.
	4. **Add Relevant Legal Context**: Mention applicable jurisdictions, laws, or precedents if relevant or implied.
	5. **Avoid Hallucination**: If you are not sure, simply state that you are not sure. Do not introduce information that isn't present or implied by the original query.
	{f"6. **Special Note**: {hallucination_reason}" if hallucination else ""}
    {f"6. **Special Note**: {quality_reason}" if quality else ""}
	{f"6. **Special Note**: {compliance_reason}" if compliance else ""}
	Lastly. **Format for Embedding Optimization**: Make the prompt concise, rich in keywords, and structured to maximize cosine similarity scores.

	---

	### ✅ **Example Transformations**
	**User Query:**  
	"What are the rights of employees in Singapore regarding wrongful dismissal?"

	**Optimized Retrieval Prompts:**  
	- "Singapore employment law: wrongful dismissal rights, legal protections, case precedents."  
	- "Retrieve statutes, case law, and regulations on employee termination and wrongful dismissal in Singapore."

	---

	### ✅ **Your Task**
	Rewrite the following user query into a search-optimized prompt for vector retrieval.

	**User Query:**  
	{query}

	---

	### ✅ **Optimized Retrieval Prompt:**  
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
    response = structured_output_parser.invoke(
        [HumanMessage(content=completeness_prompt)]
    )

    return response
