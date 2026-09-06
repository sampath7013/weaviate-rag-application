from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    weaviate_collection: str = "DocumentChunk"

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()