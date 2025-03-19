from langchain.schema import HumanMessage


def summarise_document_node(state):
    model = state["model"]
    user_context = state.get("user_context", None)
    intent = state["intent"]

    prompt = f"""
    You are a legal AI assistant. Your task is to generate a concise and informative summary of the user's context (such as a document or conversation). The summary must focus on the parts that are **most relevant** to the user's **intent**, while ensuring that **critical information is not lost**.

    ### Guidelines:
    - Maintain all key facts, details, and important information that would help address the intent.
    - Eliminate unnecessary or repetitive details.
    - Be clear, professional, and accurate.
    - If the intent involves answering questions, emphasize the facts that would help answer those questions.
    - If the intent involves summarizing the document itself, focus on its main points and conclusions.

    ---

    ### **Intent**:
    {intent}

    ---

    ### **User Context (Document/Information)**:
    {user_context}

    ---

    ### **Output**:
    Provide a clear and concise summary below.
    """

    # Call the model to generate the summary
    summarised_output = model.invoke([HumanMessage(content=prompt)])

    # Update the state with the summary
    state["user_context"] = summarised_output.content

    return state
