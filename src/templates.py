from typing import Literal
from pydantic import BaseModel, Field
from src.constant import Routing


class ComplexityRank(BaseModel):
    complexity: Literal[
        Routing.COMPLEXITY_LOW, Routing.COMPLEXITY_MEDIUM, Routing.COMPLEXITY_UNRELATED
    ] = Field(
        description="Rank the complexity of the query as LOW, MEDIUM, or UNRELATED based on the available knowledge in the vectorstore."
    )


class ResponseSufficency(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the response provided by the AI is sufficient for the query."
    )
    missing_queries: list[str] = Field(
        description="List of additional retrieval queries if the response is insufficient."
    )


class HallucinationGrader(BaseModel):
    hallucination: bool = Field(description="Whether the response is hallucinating.")
    reason: str = Field(
        description="Brief explanation of the hallucination assessment."
    )


class QualityGrader(BaseModel):
    relevance: bool = Field(
        description="Whether the response is relevant to the query."
    )
    coherence: bool = Field(description="Whether the response is coherent.")
    completeness: bool = Field(description="Whether the response is complete.")
    reason: str = Field(description="Brief explanation of the quality assessment.")


class ComplianceGrader(BaseModel):
    compliance: bool = Field(
        description="Whether the response is compliant with the query."
    )
    reason: str = Field(description="Brief explanation of the compliance assessment.")


class IntentIdentification(BaseModel):
    intent_type: str
    intent: str
