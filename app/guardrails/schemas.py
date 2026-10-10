
# app/guardrails/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


class GuardDecision(BaseModel):
    category: Literal[
        "in_scope",
        "out_of_scope",
        "jailbreak",
        "ambiguous",
        "greeting",
        "capabilities",
        "farewell",
    ]

    action: Literal["allow", "block", "review"]

    reason: str = Field(
        description="Short explanation for the classification."
    )