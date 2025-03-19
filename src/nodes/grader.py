from src.templates import (
    HallucinationGrader,
    QualityGrader,
    ComplianceGrader,
    IntentIdentification,
)
from langchain.schema import HumanMessage


def intent_identification_node(state):
    model = state["model"]
    query = state["query"]

    prompt = f"""
    You are a helpful and precise legal AI assistant. Your task is to identify the intent behind a user's query. Think carefully and follow these instructions step by step:

    1. Analyze the query from a legal assistant's point of view.
    2. Determine the **intent** of the user's query. Possible intents include (but are not limited to): 
    - "qa" (the user is asking a question and expects an answer)
    - "summarise" (the user wants to summarize a document or text)
    - "search" (the user is asking you to find information)
    - "chat" (the user is making small talk or general conversation)
    3. If the user is asking about a document or mentions handling a document (e.g., uploading, summarizing, or reviewing it), you must include "summarise" in the **intent_type** list.
    4. Think carefully about multiple intents. For example, if the user wants to summarize a document and ask a question about it, include both "summarise" and "qa".
    5. Provide a brief explanation of the **intent** in plain language.

    ⚠️ IMPORTANT: Respond only with a valid JSON object in the following format:
    {{
        "intent_type": "summarise", "qa"
        "intent": "The user wants to summarize the document and then ask a question about its content."  // A clear, short explanation
    }}

    Here is the user's query:
    "{query}"

    Respond with JSON only:
    """

    # Assuming you have a pydantic schema called intent_identification_template
    structured_output_parser = model.with_structured_output(IntentIdentification)

    # Send the prompt as a HumanMessage (like a user message)
    decision_response = structured_output_parser.invoke([HumanMessage(content=prompt)])

    # Assuming decision_response matches your intent_identification_template structure
    state["intent_type"] = decision_response.intent_type
    state["intent"] = decision_response.intent

    return state


def grade_hallucination_node(state):
    """
    Checks if the response contains hallucinated or fabricated info.
    We'll ask the model: "Does this text contain hallucinations? (YES/NO)"
    Then set 'hallucination' to True (YES) or False (NO).
    """
    model = state["model"]
    response = state["response"].content
    user_context = state["user_context"]
    system_context = state["system_context"]

    check_prompt = f"""
You are an **expert AI fact checker with legal domain expertise**, specializing in detecting hallucinated, fabricated, or inaccurate information in legal advice and documents.

---

### ✅ **Objective**
Determine whether the provided **Statement** contains any fabricated, unverified, or inaccurate legal information. Use only the **Retrieved Legal Context** and **User Uploaded Context** as your source of truth.

---

### ✅ **Source Context for Fact Checking**
{f"**Retrieved Legal Context (Vector Store)**:\n{system_context}\n" if system_context else ""}
{f"**User Uploaded Context**:\n{user_context}\n" if user_context else ""}

---

### ✅ **Statement to Verify**
{response}

---

### ✅ **Fact-Checking Instructions**

1. **Strict Verification**  
   - Cross-check each claim in the Statement against the provided contexts.  
   - Do **not** use external knowledge or make assumptions.  
   - If a claim is not directly supported by the provided context, it must be considered **hallucinated or unverified**.

2. **Determine Hallucination Status**  
   - If **all** claims in the Statement are fully supported by the context, return:  
     `"hallucination": false`  
   - If **any** claim is unsupported, unverifiable, fabricated, or irrelevant based on the provided context, return:  
     `"hallucination": true`

3. **Provide a Clear Reason**  
   - If hallucination is **true**, briefly explain **which part** of the Statement is inaccurate or unsupported, and why.  
   - If hallucination is **false**, confirm the Statement is fully supported by the provided context.

---

### ✅ **Output Format (JSON)**

```json
{{
  "hallucination": true | false,
  "reason": "Brief explanation (one sentence)."
}}
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
    intent = state["intent"]
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

    User Intent:
    {intent}

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
