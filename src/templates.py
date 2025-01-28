from typing import Literal
from pydantic import BaseModel, Field
from src.constant import Routing

class ComplexityRank(BaseModel):
    complexity: Literal[
        Routing.COMPLEXITY_LOW, Routing.COMPLEXITY_MEDIUM, Routing.COMPLEXITY_HIGH
    ] = Field(
        description="Rank the complexity of the query as LOW, MEDIUM, or HIGH based on the available knowledge in the vectorstore."
    )