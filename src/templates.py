from typing import Literal
from pydantic import BaseModel, Field
from src.constant import Routing

class ComplexityRank(BaseModel):
    complexity: Literal[
        Routing.COMPLEXITY_LOW, Routing.COMPLEXITY_MEDIUM, Routing.COMPLEXITY_HIGH
    ] = Field(
        description="Rank the complexity of the query as LOW, MEDIUM, or HIGH based on the available knowledge in the vectorstore."
    )

class ResponseSufficency(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the response provided by the AI is sufficient for the query."
    )
    missing_queries: list[str] = Field(
        description="List of additional retrieval queries if the response is insufficient."
    )