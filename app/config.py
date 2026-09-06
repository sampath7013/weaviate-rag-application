from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"

    openai_api_key: str

    # Local Weaviate
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # Weaviate Cloud
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None

    weaviate_collection: str = "DocumentChunk"

    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-5.6-luna"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()