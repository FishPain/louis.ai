from src.templates import HallucinationGrader, QualityGrader, ComplianceGrader
from langchain.schema import HumanMessage


def grade_hallucination_node(state):
    """
    Checks if the response contains hallucinated or fabricated info.
    We'll ask the model: "Does this text contain hallucinations? (YES/NO)"
    Then set 'hallucination' to True (YES) or False (NO).
    """
    model = state["model"]
    response = state["response"]

    check_prompt = f"""
    **Role:**  
    You are an **expert AI fact checker** specializing in verifying the accuracy of statements. Your objective is to detect fabricated, unverified, or misleading information (hallucinations) in the given text and provide a reason for your decision.

    ---

    ### **Task Instructions**
    1. **Analyze the Given Text:**
        - Carefully examine the provided statement(s).
        - Compare each claim to verifiable knowledge sources (e.g., structured databases, authoritative documents, or web search results).

    2. **Determine Factuality:**
        - If all claims in the text are factually correct, based on known data, return:  
            **"hallucination": "NO"** (No hallucinations detected).  
        - If any claim contains a fabricated, unverified, or irrelevant statement, return:  
            **"hallucination": "YES"** (Hallucinations detected).

    3. **Provide a Reason:**
        - If **hallucination = "YES"**, briefly explain why the statement is inaccurate or unverifiable.
        - If **hallucination = "NO"**, confirm that the statement aligns with verified knowledge.

    4. **Output Format:**
        - **Return the output in an easily parsable format:** 
        "hallucination": "YES" | "NO", "reason": "Brief explanation of factuality assessment."
        - **Ensure the reason is concise** (one sentence is sufficient).

    ---

    ### **Statement**
    {response}

    ---

    ### **Output Format**
    "hallucination": "True" | "False", "reason": "Brief explanation of factuality assessment."
    """

    structured_output_parser = model.with_structured_output(HallucinationGrader)
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=check_prompt)]
    )
    state["hallucination"] = decision_response.hallucination
    state["hallucination_reason"] = decision_response.reason
    return state


def grade_quality_node(state):
    """
    Determines if the response meets a certain 'quality' threshold.
    Uses a model to assess relevance, coherence, and completeness.
    Sets 'quality' to True if all criteria are met; otherwise, False.
    Also provides reasoning.
    """
    model = state["model"]
    query = state["query"]
    response = state["response"]

    check_prompt = f"""
    You are an expert AI legal evaluator specializing in grading the quality of AI-generated legal responses.
    Your objective is to evaluate the following response based on legal relevance, coherence, and completeness.

    ---

    ### Task Instructions
    1. Analyze the Given Response:
    - Examine the response for clarity, accuracy, and completeness.
    - Ensure the response directly answers the user’s query.
    - Identify any vague, misleading, or incomplete content.

    2. Grade the Response Based on the Following Criteria:
    - relevance (Boolean): Does the response fully address the user’s query?
    - coherence (Boolean): Is the response logically structured and easy to understand?
    - completeness (Boolean): Does the response provide enough detail to be useful?

    ---

    ### Inputs:
    User Query:
    {query}

    Generated Output:
    {response}

    ---

    ### Output Format (in JSON):
    {{
    "relevance": true/false,
    "coherence": true/false,
    "completeness": true/false,
    "reason": "Brief explanation of the grading decision."
    }}
    """

    structured_output_parser = model.with_structured_output(QualityGrader)
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=check_prompt)]
    )

    # Assume decision_response is an object or dict with the following fields
    relevance = decision_response.relevance
    coherence = decision_response.coherence
    completeness = decision_response.completeness

    # Quality is True only if all three criteria are True
    state["quality"] = relevance and coherence and completeness
    state["quality_reason"] = decision_response.reason

    return state


def grade_compliance_node(state):
    """
    Checks if the response meets certain compliance criteria (e.g.,
    no disallowed content, no policy violations).
    We'll ask the model: "Is this text compliant? (YES/NO)"
    Then set 'compliance' to True (YES) or False (NO).
    """

    model = state["model"]
    response = state["response"]

    check_prompt = f"""

    **Role:**  
    You are an **expert AI legal compliance evaluator** specializing in **Singaporean law**. 
    Your objective is to determine whether a given AI-generated legal response is **legally valid within Singapore’s legal framework**.
    You must assess the response in relation to **Singapore’s Constitution, statutory laws, case precedents, and relevant regulations** to ensure **legal compliance**.

    ---

    ### **Task Instructions**
    1. **Analyze the Given Legal Response:**
    - Ensure that the response **aligns with Singaporean laws, regulations, and case precedents**.
    - Verify if the response is **factually and legally correct** under **Singapore’s legal framework**.
    - Cross-check the response with **the given context** to determine its applicability.
    - Identify any **misleading, outdated, or jurisdictionally incorrect statements**.

    2. **Determine Singaporean Legal Validity:**
    - If the response is **fully compliant** with Singaporean law and aligns with the context, return:  
        **"valid": "YES"** (Fully compliant).  
    - If the response **contains legally invalid, misleading, or non-Singaporean legal principles**, or does not align with the context, return:  
        **"valid": "NO"** (Not compliant).

    3. **Provide a Reason:**
    - If **valid = "NO"**, briefly explain the legal issue (e.g., misalignment with Singaporean law, incorrect legal principle, outdated information).  
    - If **valid = "YES"**, confirm that the response follows Singaporean law.

    4. **Output Format:**
        - **Return the output in an easily parsable format:** 
        "valid": "YES" | "NO", "reason": "Brief explanation of legal compliance or non-compliance."
        - **Ensure the reason is concise** (one sentence is sufficient).
    ---

    ### Inputs

    AI Generated Response:
    {response}

    ---

    ### **Output Format**
    "valid": "True" | "False", "reason": "Brief explanation of legal compliance or non-compliance."

    """

    structured_output_parser = model.with_structured_output(ComplianceGrader)
    decision_response = structured_output_parser.invoke(
        [HumanMessage(content=check_prompt)]
    )
    state["compliance"] = decision_response.compliance
    state["compliance_reason"] = decision_response.reason
    return state
