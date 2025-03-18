from langchain.schema import HumanMessage

from src.constant import Routing
from src.templates import ComplexityRank

from langchain.schema import HumanMessage
from src.constant import Routing
from src.templates import ComplexityRank


def complexity_scoring_node(state):
    """
    Uses an LLM to rank the complexity of the query based on what is already known in the vector store.
    Returns 'LOW', 'MEDIUM', 'HIGH' complexity, or 'UNRELATED' if it's out of scope.
    """
    query = state["query"]
    model = state["model"]
    vectorstore_summary = state["vectorstore_summary"]
    decision_prompt = f"""
    You are an advanced AI legal reasoning system. Your role is to **evaluate the complexity of a legal query** and determine how well it can be answered, based on the knowledge in the vector store and the capabilities of your own reasoning.

    ---
    
    ### ✅ **Your Task**
	You will receive:
	1. A **User Query** (a legal question).
	2. A **Vector Store Summary** (describing what legal information the vector store contains).

	Your job is to **evaluate** whether the query:
	- Can be answered directly by you (GPT) without needing the vector store.
	- Requires additional information from the vector store.
	- Is unrelated to the legal field entirely.

	---

	### ✅ **How to Rank Complexity**
	Choose **one** complexity level based on the criteria below.

	#### ✅ {Routing.COMPLEXITY_LOW} (LOW)
	- The query is **simple**, **direct**, and can be fully answered **by GPT itself**, using your own knowledge and reasoning.
	- You **do not need** the vector store to answer it.
	- Example: "What is the minimum annual leave entitlement under Singapore law?"  
	➡️ This is a **fact-based question** that GPT can answer without extra help.

	#### 🟧 {Routing.COMPLEXITY_MEDIUM} (MEDIUM)
	- The query requires **additional information** that **GPT does not have**, and must retrieve from the **vector store**.
	- Example situations:  
	- The query requests **specific case law** or **legal precedents**, which are only found in the vector store.  
	- The query relates to **custom legal interpretations** or **specialized documents** stored in the vector store.  
	➡️ GPT cannot provide a **complete** answer without accessing the vector store.

	#### 🚫 {Routing.COMPLEXITY_UNRELATED} (UNRELATED)
	- The query is **completely unrelated** to the **field of law or legal practice**.
	- Example:  
	- Asking about **weather**, **technology**, or **medical** topics.  
	- Requesting information about **Malaysia law** when the vector store only contains **Singapore law**.

	---

	### ✅ **Clarifications**
	- **Case Law Queries:**  
	If the user asks for **case law** or **legal precedents**, and GPT does not have direct access to them, rank the complexity as **MEDIUM**, **not** LOW.  
	➡️ Even if GPT can provide **general legal principles**, without specific cases it needs vector store support.

	- **Avoid UNRELATED unless it's truly out-of-scope:**  
	➡️ Only use UNRELATED if the query is not about **law** or not covered by your **legal expertise** at all.

	---

	### ✅ **Vector Store Summary**
	{vectorstore_summary}

	---

	### ✅ **User Query**
	{query}

	---

	### ✅ **Final Decision**
	Choose **one** complexity ranking from the following list:  
	**{Routing.COMPLEXITY_LOW} / {Routing.COMPLEXITY_MEDIUM} / {Routing.COMPLEXITY_UNRELATED}**

	Respond with the **selected complexity level only**.
	"""
    structured_output_parser = model.with_structured_output(ComplexityRank)
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=decision_prompt)]
    )
    state["complexity"] = decision_response.complexity
    return state
