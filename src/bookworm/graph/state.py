from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from bookworm.graph.schemas import IntentLabel


class BookResult(TypedDict):
    book_id:     str
    title:       str
    author:      str
    year:        int | None
    genres:      str | None
    description: str | None


class PendingLendingAction(TypedDict):
    tool_name: str
    arguments: dict


class MainState(TypedDict):
    user_id:             str
    messages:            Annotated[list[AnyMessage], add_messages]
    
    intent:              IntentLabel
    task_plan:           list[dict]
    search_results:      NotRequired[list[BookResult]]
    lending_result:      NotRequired[str]

    response_attempts:   NotRequired[int]
    evaluation_feedback: NotRequired[str]
    revision_requested:  NotRequired[bool]


class RagState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    search_task_description: str

    retrieval_plan:     NotRequired[dict]
    semantic_results:   NotRequired[list[BookResult]]
    lexical_results:    NotRequired[list[BookResult]]
    fused_results:      NotRequired[list[BookResult]]

    retrieval_attempts: NotRequired[int]
    retrieval_feedback: NotRequired[str]
    retry_retrieval:    NotRequired[bool]


class LendingState(TypedDict):
    user_id:                  str
    messages:                 Annotated[list[AnyMessage], add_messages]
    lending_task_description: str
    search_results:           list[BookResult]

    pending_lending_action:   NotRequired[PendingLendingAction | None]
    approval_granted:         NotRequired[bool]
    lending_result:           NotRequired[str]