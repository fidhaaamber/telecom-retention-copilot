from typing import Any
from pydantic import BaseModel, Field

class CustomerRecord(BaseModel):
    customer: dict[str, Any] = Field(...)

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: str
    decision_threshold: float
    drivers: list[dict[str, Any]]

class RecommendationResponse(BaseModel):
    recommended_action: str
    rationale: str
    observed_facts: list[str]
    limitations: str
    requires_analyst_approval: bool = True
