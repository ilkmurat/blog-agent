from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def last_is_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be from the user")
        return self


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: Literal["ok", "error"]
    mcp: Literal["ok", "error"]
    model: str
    langfuse: Literal["enabled", "disabled"]


class Error(BaseModel):
    detail: str
