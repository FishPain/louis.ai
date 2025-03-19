from langchain.schema import HumanMessage
from src.templates import ResponseSufficency


def create_retrieval_prompt_node(state):
    """
    Node: Create an optimized retrieval prompt for similarity search in the vector store.
    """

    query = state["query"]
    model = state["model"]
    intent = state["intent"]
    user_context = state.get("user_context", None)
    hallucination = state.get("hallucination", False)
    hallucination_reason = state.get("hallucination_reason", "")
    quality = state.get("quality", False)
    quality_reason = state.get("quality_reason", "")
    compliance = state.get("compliance", False)
    compliance_reason = state.get("compliance_reason", "")

    prompt = f"""
You are a **legal AI assistant** specializing in **optimizing legal queries for similarity-based vector retrieval** from a legal knowledge base.

---

### ✅ **Objective**
Rewrite the user query into a **concise, keyword-rich prompt**, designed to **maximize relevance** in vector similarity search. The goal is to retrieve the **most accurate and relevant legal documents** related to the query.

---

### ✅ **Instructions**
Follow these steps carefully when optimizing the retrieval prompt:

1. **Extract Key Legal Concepts**  
   - Identify and prioritize critical legal terms, statutes, jurisdictions, case law references, and legal processes mentioned or implied in the query.
   
2. **Clarify Jurisdictions and Legal Domains**  
   - If jurisdiction (e.g., Singapore law) is known or implied, explicitly mention it in the optimized prompt.

3. **Expand with Synonyms and Legal Variations**  
   - Include common variations, alternative legal phrases, or synonyms that improve matching in vector search.

4. **Simplify and Disambiguate**  
   - Remove vague, non-legal, or ambiguous language. Use **clear, specific legal terminology**.

5. **Contextualize the Query**  
   - Leverage any provided **User Uploaded Context** to improve precision (if relevant).

6. **Avoid Hallucination**  
   - Do **not** introduce information not present or clearly implied in the query or context.
   {f"- ⚠️ Special Note: {hallucination_reason}" if hallucination else ""}
   {f"- ⚠️ Special Note: {quality_reason}" if quality else ""}
   {f"- ⚠️ Special Note: {compliance_reason}" if compliance else ""}

7. **Format for Embedding Optimization**  
   - Write a **concise, keyword-dense prompt**. Avoid long sentences. Focus on **key terms** separated by commas or structured phrases to boost cosine similarity scores.

---

### ✅ **Example Transformations**

**User Query:**  
"What are the rights of employees in Singapore regarding wrongful dismissal?"

**Optimized Retrieval Prompts:**  
- "Singapore employment law, wrongful dismissal, employee rights, case law precedents."  
- "Legal protections, employee termination, wrongful dismissal claims, Singapore statutes and case law."

---

### ✅ **Provided Information**

{f"**User Uploaded Context**:\n{user_context}\n\n" if user_context else ""}
**User Intent:**  
{intent}

**Original User Query:**  
{query}

---

### ✅ **Your Task**  
Based on the above instructions, rewrite the user's query as an **optimized retrieval prompt** that will maximize relevance and accuracy during vector-based search.

---

### ✅ **Optimized Retrieval Prompt:**  
"""

    response = model.invoke([HumanMessage(content=prompt)])
    state["response"] = response
    return state


def check_completeness_with_llm(state):
    model, query = state["model"], state["query"]
    user_context = state.get("user_context", None)
    system_context = state.get("system_context", None)
    """
    Uses the LLM to determine whether the retrieved legal context is sufficient 
    or if additional retrieval is necessary.
    """
    completeness_prompt = f"""
You are a **highly skilled legal AI auditor**. Your role is to **critically assess the completeness and sufficiency** of the retrieved legal context for answering a user’s query.

Your evaluation must be **thorough, precise, and strict**. Assume that any missing legal reference or incomplete citation could result in an inaccurate or incomplete legal response.

---

### ✅ **User Query**  
{query}
---
{f"### ✅ **User Uploaded Context**\n{user_context}\n---" if user_context else ""}
{f"### ✅ **Retrieved Legal Context**  \n{system_context}\n---" if system_context else ""}

### ✅ **Your Task**

1. **Evaluate Completeness**  
   - Determine if the retrieved legal context provides **all necessary legal references**, including statutes, case law, sections, sub-sections, and definitions, to answer the query **fully and accurately**.  
   - **Strictly verify** that all referenced materials in the retrieved context are **fully present and explained**.  
     - Example: If the retrieved context mentions "Section 14(2)" but does **not** include the content of Section 14(2), consider the retrieval **incomplete**.
   - Confirm whether the context covers **all legal principles or doctrines** implied or necessary to answer the query.

2. **Identify Missing Information**  
   - List **specific legal terms, sections, cases, or topics** that need to be retrieved to complete the answer.  
   - Base this only on **gaps found in the Retrieved Legal Context**, not general knowledge.

---

### ✅ **Evaluation Criteria**
- **Sufficient** if:  
  - All cited statutes, cases, sections, and key legal concepts are fully provided and explained in the retrieved context.  
  - There is **no ambiguity** or **missing cross-references** needed to answer the query.

- **Insufficient** if:  
  - Any cited law, clause, section, or concept is referenced but **not fully included** in the retrieved context.  
  - Key legal principles necessary to answer the query are **absent**.

---

### ✅ **Response Format (Strict JSON Only)**

```json
{{
  "is_sufficient": true / false,
  "missing_queries": [
    "Section X details",
    "Case law on [specific legal concept]",
    "Statutory definition of [term]"
  ]
}}
"""
    structured_output_parser = model.with_structured_output(ResponseSufficency)
    response = structured_output_parser.invoke(
        [HumanMessage(content=completeness_prompt)]
    )

    return response
