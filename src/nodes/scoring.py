from langchain.schema import HumanMessage

from src.constant import Routing
from src.templates import ComplexityRank

from langchain.schema import HumanMessage
from src.constant import Routing
from src.templates import ComplexityRank


def complexity_scoring_node(state):
    """
    Uses an LLM to rank the complexity of the query based on what is already known
    in the vector store and any user-uploaded context.
    Returns 'LOW', 'MEDIUM', 'HIGH' complexity, or 'UNRELATED' if it's out of scope.
    """
    query = state["query"]
    model = state["model"]
    user_context = state.get("user_context", None)
    vectorstore_summary = state["vectorstore_summary"]

    decision_prompt = f"""
You are an advanced AI legal reasoning system. Your role is to **evaluate the complexity of a legal query** and determine how well it can be answered, based on:
1. Your own general legal knowledge and reasoning.
2. The information available in the **Vector Store Summary**.
3. Any additional **User Uploaded Context**, such as documents or notes provided by the user.

---

### ✅ **Your Task**
You will receive:
1. A **User Query** (a legal question).
2. A **Vector Store Summary** (describing what legal information the vector store contains).
3. A **User Uploaded Context** (a document or additional details provided by the user).

Your job is to **evaluate** whether the query:
- Can be answered directly by you (GPT) using your own reasoning or the user-uploaded context.
- Requires additional information from the vector store.
- Is unrelated to the legal field entirely.

---

### ✅ **How to Rank Complexity**
Choose **one** complexity level based on the criteria below.

#### ✅ LOW Complexity (**{Routing.COMPLEXITY_LOW}**)
- The query is **simple**, **direct**, and can be fully answered **by GPT itself** or **using the provided User Uploaded Context**, without needing the vector store.
- Example: "What is the minimum annual leave entitlement under Singapore law?"  
➡️ This is a **fact-based question** that GPT can answer without extra help.

#### 🟧 MEDIUM Complexity (**{Routing.COMPLEXITY_MEDIUM}**)
- The query requires **additional information** from the **vector store**, even after reviewing the **User Uploaded Context**.
- Example:  
  - The query requests **specific case law**, **legal precedents**, or **custom legal interpretations** not found in the user context.
  - The uploaded context is **insufficient** to fully answer the query.
➡️ GPT cannot provide a **complete** answer without accessing the vector store.

#### 🚫 UNRELATED (**{Routing.COMPLEXITY_UNRELATED}**)
- The query is **completely unrelated** to the **field of law or legal practice**.
- Example:  
  - Asking about **weather**, **technology**, or **medical** topics.  
  - Requesting information about **Malaysia law**, when both the user context and vector store only cover **Singapore law**.

---

### ✅ **Clarifications**
- **Case Law Queries:**  
  If the user asks for **case law** or **legal precedents**, and neither GPT nor the User Uploaded Context provides this information, rank it as **MEDIUM**.
  
- **Use User Uploaded Context First:**  
  If the user's uploaded document provides enough information to answer the query completely, you may assign **LOW** complexity (even if the vector store isn’t needed).

- **Avoid UNRELATED unless it's truly out-of-scope:**  
  Only use UNRELATED if the query is not about **law** or **outside the provided knowledge areas**.

---

### ✅ **Vector Store Summary**
{vectorstore_summary}

---

{
    f"### ✅ **User Uploaded Context**\n{user_context}\n---" if user_context else ""
}

### ✅ **User Query**
{query}

---

### ✅ **Final Decision**
Choose **one** complexity ranking from the following list (no explanations):  
**{Routing.COMPLEXITY_LOW} / {Routing.COMPLEXITY_MEDIUM} / {Routing.COMPLEXITY_UNRELATED}**

Respond with the **selected complexity level only**.
"""

    structured_output_parser = model.with_structured_output(ComplexityRank)
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=decision_prompt)]
    )
    state["complexity"] = decision_response.complexity
    return state
