from __future__ import annotations

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    rev_util: float = Field(ge=0, description="Revolving utilization")
    age: float = Field(ge=18, le=120)
    late_30_59: int = Field(ge=0)
    debt_ratio: float = Field(ge=0)
    monthly_inc: float = Field(ge=0)
    open_credit: int = Field(ge=0)
    late_90: int = Field(ge=0)
    real_estate: int = Field(ge=0)
    late_60_89: int = Field(ge=0)
    dependents: int = Field(ge=0)


class Reason(BaseModel):
    code: str
    description: str
    contribution: float


class ScoreResponse(BaseModel):
    pd: float
    decision: str
    threshold: float
    reasons: list[Reason]


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
