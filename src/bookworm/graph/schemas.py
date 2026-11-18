from typing import Literal
from pydantic import BaseModel, Field


IntentLabel = Literal[
    "catalogue_search",
    "reservation",
    "cancellation",
    "renewal",
    "combined",
    "general",
    "unknown",
]


class Intent(BaseModel):
    intent: IntentLabel = Field(description="The classification of the user's latest request.")


class PlannedTask(BaseModel):
    agent: Literal["search", "lending"] = Field(description="The agent responsible for executing this task.")
    description: str = Field(description="A concise description of what the agent must accomplish.")


class TaskList(BaseModel):
    tasks: list[PlannedTask] = Field(description="Ordered list of tasks required to fulfil the user's request.")


class LendingAction(BaseModel):
    """A lending operation selected by the Lending Agent."""

    operation: Literal[
        "check_availability",
        "make_reservation",
        "cancel_reservation",
        "renew_loan",
        "no_action",
    ]

    book_id: str | None = Field(
        default=None,
        description="Book ID for availability, reservation, or cancellation.",
    )

    loan_id: str | None = Field(
        default=None,
        description="Loan ID for renewal.",
    )

    reason: str = Field(
        default="",
        description="Explanation if no action can be performed.",
    )


class Evaluation(BaseModel):
    """The Evaluator's verdict on a generated response."""

    passed:  bool = Field(description="True if the response is appropriate and factually correct.")
    feedback: str = Field(
        default="",
        description="If the response failed, a concise explanation of what must be fixed.",
    )


class MetadataFilter(BaseModel):
    """Structured constraints applied to the catalogue search."""

    author:    str | None = Field(default=None, description="Author name, only if the user explicitly names one.")
    genre:     str | None = Field(default=None, description="Genre, only if the user explicitly asks for one.")
    year_from: int | None = Field(default=None, description="Earliest publication year (inclusive).")
    year_to:   int | None = Field(default=None, description="Latest publication year (inclusive).")


class RetrievalPlan(BaseModel):
    """Search queries produced by the Query Processor."""

    keyword_query: str = Field(
        default="",
        description="Short keyword query for lexical search: titles, author names, distinctive terms. Empty if not useful.",
    )
    semantic_query: str = Field(
        default="",
        description="Natural-language description of the wanted books for semantic search. Empty if not useful.",
    )
    metadata_filter: MetadataFilter = Field(
        default_factory=MetadataFilter,
        description="Explicit constraints from the user's request.",
    )

