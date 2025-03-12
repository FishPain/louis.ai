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

    # Define the improved decision prompt
    decision_prompt = f"""
	You are an advanced AI legal reasoning system. Your role is to **evaluate the complexity of a legal query** relative to the knowledge contained in the vector store.

	---

	### ✅ **Your Task**
	1. **Understand the User Query**
	- What legal information is the user requesting?
	- Does it concern a simple fact, require legal interpretation, or involve multiple legal issues?

	2. **Analyze the Vector Store Knowledge**
	- Review the **Vector Store Summary** below.
	- Determine if the query is **fully answered**, **partially answered**, or **unrelated** to the knowledge it contains.

	---

	### ✅ **How to Rank Complexity**
	Select **one** of the following complexity levels:

	- **{Routing.COMPLEXITY_UNRELATED} (UNRELATED)**  
	> The query is **not relevant** to the topics covered by the vector store.  
	> Example: The vector store contains information about **Singapore employment law**, but the user is asking about **EU data privacy regulations**.  
	> If the query falls outside the scope of the available content, rank it as UNRELATED.

	- **{Routing.COMPLEXITY_LOW} (LOW)**  
	> The query is directly and fully answered by information in the vector store.  
	> No extra reasoning or external information is required.

	- **{Routing.COMPLEXITY_MEDIUM} (MEDIUM)**  
	> Some relevant information is available in the vector store.  
	> Minor reasoning or external context may be needed to complete the answer.

	- **{Routing.COMPLEXITY_HIGH} (HIGH)**  
	> The vector store lacks sufficient information to answer the query fully.  
	> Significant external research or legal analysis would be required.

	---

	### ✅ **Vector Store Summary**
	{vectorstore_summary}

	---

	### ✅ **User Query**
	{query}

	---

	### ✅ **Final Decision**
	Choose one complexity ranking from the following list:  
	**{Routing.COMPLEXITY_UNRELATED} / {Routing.COMPLEXITY_LOW} / {Routing.COMPLEXITY_MEDIUM} / {Routing.COMPLEXITY_HIGH}**

	Respond with the selected complexity level only.
	"""

    # Set up the structured output parser
    structured_output_parser = model.with_structured_output(ComplexityRank)

    # Invoke the model with the prompt
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=decision_prompt)]
    )

    state["complexity"] = decision_response.complexity
    return state
