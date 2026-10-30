from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_num_ctx: int = 8192
    # Thinking models (qwen3): False skips the reasoning phase for faster chat answers; None = model default.
    ollama_reasoning: bool | None = False
    temperature: float = 0.2

    blog_api_url: str = "http://localhost:8000"
    mcp_url: str = "http://localhost:8000/mcp/"

    # Langfuse tracing is enabled only when both keys are set (LANGFUSE_BASE_URL selects the region/host).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # The chat is public: protect the local GPU.
    rate_limit_per_minute: int = 10
    max_concurrent_chats: int = 2

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
