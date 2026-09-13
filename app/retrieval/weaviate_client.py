from __future__ import annotations

import atexit
import logging
import threading

import weaviate

from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
)
from weaviate.classes.init import (
    AdditionalConfig,
    Auth,
    Timeout,
)

from app.config import settings


logger = logging.getLogger(__name__)


# ============================================================
# Shared retrieval client
# ============================================================

_shared_client = None

_shared_client_lock = threading.Lock()


# ============================================================
# Timeout configuration
# ============================================================

def _build_timeout_config() -> AdditionalConfig:
    """
    Create Weaviate timeout configuration.

    init:
        Initial connection / health-check timeout.

    query:
        Query timeout.

    insert:
        Ingestion timeout.
    """

    return AdditionalConfig(
        timeout=Timeout(
            init=30,
            query=60,
            insert=120,
        )
    )


# ============================================================
# Client creation
# ============================================================

def _create_weaviate_client():
    """
    Create a NEW Weaviate client.

    Supports:
        - Weaviate Cloud in production
        - local Docker Weaviate in development
    """

    timeout_config = (
        _build_timeout_config()
    )

    # ========================================================
    # Production - Weaviate Cloud
    # ========================================================

    if (
        settings.app_env.lower()
        == "production"
    ):

        if not settings.weaviate_url:
            raise ValueError(
                "WEAVIATE_URL is required in production."
            )

        if not settings.weaviate_api_key:
            raise ValueError(
                "WEAVIATE_API_KEY is required in production."
            )

        logger.info(
            "weaviate_cloud_connection_creating"
        )

        client = (
            weaviate.connect_to_weaviate_cloud(
                cluster_url=(
                    settings.weaviate_url
                ),
                auth_credentials=(
                    Auth.api_key(
                        settings.weaviate_api_key
                    )
                ),
                additional_config=(
                    timeout_config
                ),
            )
        )

        logger.info(
            "weaviate_cloud_connection_created"
        )

        return client

    # ========================================================
    # Development - Local Weaviate
    # ========================================================

    logger.info(
        (
            "weaviate_local_connection_creating | "
            "host=%s | "
            "http_port=%s | "
            "grpc_port=%s"
        ),
        settings.weaviate_host,
        settings.weaviate_http_port,
        settings.weaviate_grpc_port,
    )

    client = (
        weaviate.connect_to_local(
            host=(
                settings.weaviate_host
            ),
            port=(
                settings.weaviate_http_port
            ),
            grpc_port=(
                settings.weaviate_grpc_port
            ),
            additional_config=(
                timeout_config
            ),
        )
    )

    logger.info(
        "weaviate_local_connection_created"
    )

    return client


# ============================================================
# Short-lived client
# ============================================================

def get_weaviate_client():
    """
    Return a NEW independent Weaviate client.

    Intended for:
        - ingestion
        - schema creation
        - migration scripts
        - administrative operations

    The caller is responsible for calling:

        client.close()
    """

    return _create_weaviate_client()


# ============================================================
# Shared retrieval client
# ============================================================

def get_shared_weaviate_client():
    """
    Return one process-level shared Weaviate client.

    Retrieval should use this function so Cloud connection
    setup and gRPC health-check overhead are not paid for
    every question.

    Lifecycle:

        first retrieval
            -> create client

        subsequent retrieval
            -> reuse client

        process/app shutdown
            -> close client
    """

    global _shared_client

    # Fast path after initialization.
    if _shared_client is not None:
        return _shared_client

    # Prevent multiple simultaneous initializations.
    with _shared_client_lock:

        if _shared_client is None:

            logger.info(
                "weaviate_shared_client_initializing"
            )

            _shared_client = (
                _create_weaviate_client()
            )

            logger.info(
                "weaviate_shared_client_initialized"
            )

    return _shared_client


# ============================================================
# Shared client shutdown
# ============================================================

def close_shared_weaviate_client() -> None:
    """
    Close the shared Weaviate retrieval client.

    Safe to call multiple times.

    Used by:
        - FastAPI lifespan shutdown
        - Python interpreter shutdown via atexit
        - evaluation scripts
        - Streamlit process shutdown
    """

    global _shared_client

    with _shared_client_lock:

        if _shared_client is None:
            return

        logger.info(
            "weaviate_shared_client_closing"
        )

        try:
            _shared_client.close()

            logger.info(
                "weaviate_shared_client_closed"
            )

        except Exception:
            logger.exception(
                "weaviate_shared_client_close_failed"
            )

        finally:
            _shared_client = None


# ============================================================
# Shared client reset
# ============================================================

def reset_shared_weaviate_client() -> None:
    """
    Reset the shared client.

    Useful if a long-lived Cloud connection becomes unhealthy.

    The next retrieval request will automatically create a
    fresh connection.
    """

    logger.warning(
        "weaviate_shared_client_reset"
    )

    close_shared_weaviate_client()


# ============================================================
# Process-level cleanup
# ============================================================

# This handles:
#
#   python -m evaluation.retrieval_diagnostics
#   python run_ingestion.py
#   Streamlit process termination
#   interactive Python shutdown
#
# close_shared_weaviate_client() is idempotent, so FastAPI can
# also explicitly close it during application shutdown.

atexit.register(
    close_shared_weaviate_client
)


# ============================================================
# Collection creation
# ============================================================

def create_collection(
    client,
) -> None:
    """
    Create the DocumentChunk collection if it does not exist.
    """

    collection_name = (
        settings.weaviate_collection
    )

    if client.collections.exists(
        collection_name
    ):

        logger.info(
            (
                "weaviate_collection_exists | "
                "collection=%s"
            ),
            collection_name,
        )

        return

    client.collections.create(
        name=collection_name,
        vector_config=(
            Configure.Vectors.self_provided()
        ),
        properties=[
            Property(
                name="text",
                data_type=DataType.TEXT,
            ),
            Property(
                name="document_name",
                data_type=DataType.TEXT,
            ),
            Property(
                name="document_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="file_hash",
                data_type=DataType.TEXT,
            ),
            Property(
                name="page_number",
                data_type=DataType.INT,
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
            ),
        ],
    )

    logger.info(
        (
            "weaviate_collection_created | "
            "collection=%s"
        ),
        collection_name,
    )


# ============================================================
# Schema migration helper
# ============================================================

def ensure_required_properties(
    client,
) -> None:
    """
    Ensure metadata properties required by the application
    exist on the collection.
    """

    collection = (
        client.collections.use(
            settings.weaviate_collection
        )
    )

    config = (
        collection.config.get()
    )

    existing_properties = {
        prop.name
        for prop in config.properties
    }

    required_properties = {
        "document_id": DataType.TEXT,
        "file_hash": DataType.TEXT,
    }

    for (
        property_name,
        data_type,
    ) in required_properties.items():

        if (
            property_name
            not in existing_properties
        ):

            collection.config.add_property(
                Property(
                    name=property_name,
                    data_type=data_type,
                )
            )

            logger.info(
                (
                    "weaviate_property_added | "
                    "collection=%s | "
                    "property=%s"
                ),
                settings.weaviate_collection,
                property_name,
            )

        else:

            logger.info(
                (
                    "weaviate_property_exists | "
                    "collection=%s | "
                    "property=%s"
                ),
                settings.weaviate_collection,
                property_name,
            )