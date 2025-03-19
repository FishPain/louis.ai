from src.templates import (
    HallucinationGrader,
    QualityGrader,
    ComplianceGrader,
    IntentIdentification,
)
from langchain.schema import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
import time


def intent_identification_node(state):
    model = state["model"]
    query = state["query"]

    prompt = f"""
    You are a helpful and precise legal AI assistant. Your task is to identify the intent behind a user's query. Think carefully and follow these instructions step by step:

    1. Analyze the query from a legal assistant's point of view.
    2. Choose **one** intent type that best describes the user's query. Possible intents are:  
    - "qa" (the user is asking a legal question and expects an answer)  
    - "summarise" (the user wants to summarize a document or text)  
    3. If the user is asking about a document or mentions handling a document (e.g., uploading, summarizing, or reviewing it), you must include "summarise" in the **intent_type** list.
    4. Think carefully about multiple intents. For example, if the user wants to summarize a document and ask a question about it, include both "summarise" and "qa".
    5. Provide a brief explanation of the **intent** in plain language.

    ⚠️ IMPORTANT: Return valid JSON ONLY in the following format:  
    ```json
    {{
        "intent_type": "summarise"/"qa"
        "intent": "The user wants to summarize the document."
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
    user_context = state.get("user_context", None)
    system_context = state.get("system_context", None)

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


def verify_hallucination_node(state):
    """
    Verifies if the AI response contains hallucinations by checking citations
    against the retrieved document metadata and (optionally) web search results.
    """
    model = state["model"]
    query = state["query"]
    intent = state["intent"]
    response = state["response"].content
    user_context = state.get("user_context", None)
    system_context = state.get("system_context", None)
    retrieved_docs = state["retrieved_docs"]  # List of Document objects

    # Extract metadata from retrieved docs
    metadata_list = [doc.metadata for doc in retrieved_docs]
    search = DuckDuckGoSearchRun()

    # Collect additional context via web search or URL validation (online context)
    online_context = ""
    # for doc in metadata_list:
    #     url = doc.get("url")
    #     if url:
    #         search_response = str(search.invoke(url))
    #         online_context += f"\n\n[Data from {url}]:\n{search_response}"
    # Create the task prompt
    prompt = f"""
You are a **legal AI fact checker and citation validator**.  
Your task is to **evaluate the factual accuracy and citation integrity** of an AI-generated legal response.

---

### ✅ Sources You Can Use for Fact Checking
1. **Retrieved Legal Context from Vector Store**:  
{system_context if system_context else 'None'}

2. **User Uploaded Context**:  
{user_context if user_context else 'None'}

3. **Retrieved Document Metadata** (includes source titles, jurisdictions, sections, URLs):  
{metadata_list}

4. **Online Context (Search Results / URL Validation)**:  
{online_context if online_context else 'None'}

---

### ✅ Objective
Determine whether the **AI Response** contains any:  
- Fabricated, unverifiable, or inaccurate legal information  
- Incorrect citations, references, or URLs  
- Misleading or incomplete claims not supported by the provided sources

---

### ✅ AI Response to Check
{response}

---

### User Query
{query}

---

### ✅ Task Instructions
1. **Fact-Check the Entire AI Response**:  
   - Verify each claim using the sources provided above.  
   - Cross-check all citations and URLs to confirm they exist and match the provided sources.

2. **Be Strict**:  
   - If **any claim or citation is unsupported**, unverifiable, or incorrect based on the provided information, label the response as hallucinated.  
   - Do **not** use external or personal knowledge beyond what is provided.

3. **Use Search as Needed**:  
   - If you can't confirm something in the retrieved context or metadata, rely on the **Online Context** you collected earlier.  
   - If it’s still unclear, assume hallucination.

---

### ✅ Expected Output Format (Strict JSON)
Respond **only** with the following JSON structure:  
```json
{{
  "hallucination": true | false,
  "reason": "Brief explanation (one sentence)."
}}

"""
    structured_output_parser = model.with_structured_output(HallucinationGrader)
    decision_response = structured_output_parser.invoke([HumanMessage(content=prompt)])
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
    Checks if the response meets ethical compliance criteria:
    - Objectivity
    - No discriminatory, racist, sexist, or offensive content
    Sets 'compliance' to True (YES) or False (NO).
    """
    model = state["model"]
    response = state[
        "response"
    ].content  # assuming response is a HumanMessage or LLM output

    # Compliance Check Prompt
    check_prompt = f"""
You are an **AI content compliance auditor**.  
Your role is to **evaluate the AI-generated response** and determine whether it complies with **ethical and policy guidelines**.

---

### ✅ Objective
Assess whether the response:
- Is **objective and neutral**, free from personal opinions or judgments.
- Does **not** contain **biased, discriminatory, racist, sexist, or offensive content**.
- Respects **diversity**, **equality**, and **inclusivity**.
- Does **not** make assumptions or promote stereotypes about any **race**, **gender**, **ethnicity**, **religion**, or **group**.

---

### ✅ Task Instructions
1. **Analyze the Response for Compliance:**
   - Look for any content that could be considered **racist**, **sexist**, **discriminatory**, or **harmful**.
   - Check whether the tone is **objective**, **respectful**, and **professional**.
   - Ensure the response avoids **biased language**, **prejudice**, or **subjective opinions**.

2. **Determine Compliance:**
   - If the response is **fully compliant**, objective, and respectful, return:  
     `"compliant": true`  
   - If the response **contains bias, discrimination, or offensive content**, return:  
     `"compliant": false`

3. **Provide a Reason:**
   - If **compliant = false**, briefly explain why (e.g., "contains gender bias", "uses inappropriate language").
   - If **compliant = true**, confirm that the response is neutral and free of bias.

---

### ✅ Input:  
AI-Generated Response:
```plaintext
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
