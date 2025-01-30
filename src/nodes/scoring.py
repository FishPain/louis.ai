from langchain.schema import HumanMessage

from src.constant import Routing
from src.templates import ComplexityRank

def complexity_scoring_node(state):
    """
    Uses an LLM to rank the complexity of the query based on what is already known in the vector store.
    Returns 'LOW', 'MEDIUM', or 'HIGH' complexity.
    """

    query = state["query"]
    model = state["model"]
    vectorstore_summary = state["vectorstore_summary"]

    # Define the improved decision prompt
    decision_prompt = f"""
    You are an advanced AI system specializing in evaluating the complexity of legal queries. 

    Your task is to **analyze the user's query** in relation to the **knowledge stored in the vector store** and determine its complexity level.

    ### **Assessment Process**
    Follow these steps:
    1. **Understand the Query**  
       - What is the user asking?  
       - Does it involve a straightforward fact, a legal interpretation, or a multi-faceted legal issue?  
       
    2. **Analyze Available Knowledge**  
       - Review the provided summary of the vector store's contents.  
       - Identify if the vector store contains **direct answers, partial information, or lacks relevant content** for this query.  

    3. **Rank the Query Complexity**  
       Choose one of the following rankings based on the level of effort required to answer the query:  

       - **LOW**: The query is fully covered by the vector store. The answer can be directly retrieved.  
       - **MEDIUM**: Some relevant information is available, but additional **reasoning or minor external context** may be required.  
       - **HIGH**: The vector store lacks sufficient information, and **external resources or case law research** will be necessary.  

    ---
    ### **Vector Store Summary**
    {vectorstore_summary}

    ### **User Query**
    {query}

    ---
    ### **Final Decision**
    Rank the complexity as one of the following:  
    **({Routing.COMPLEXITY_LOW} / {Routing.COMPLEXITY_MEDIUM} / {Routing.COMPLEXITY_HIGH})**
    """

    # Set up the structured output parser
    structured_output_parser = model.with_structured_output(ComplexityRank)

    # Invoke the model with the prompt
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=decision_prompt)]
    )
    state["complexity"] = decision_response.complexity
    return state

