import weaviate

from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
)

from weaviate.classes.init import Auth

from app.config import settings


def get_weaviate_client():

    if settings.app_env.lower() == "production":

        if not settings.weaviate_url:
            raise ValueError(
                "WEAVIATE_URL is required in production"
            )

        if not settings.weaviate_api_key:
            raise ValueError(
                "WEAVIATE_API_KEY is required in production"
            )

        return weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=Auth.api_key(
                settings.weaviate_api_key
            ),
        )

    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )


def create_collection(client):

    collection_name = settings.weaviate_collection

    if client.collections.exists(collection_name):
        print(
            f"Collection '{collection_name}' already exists."
        )
        return

    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(),
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

    print(
        f"Collection '{collection_name}' created successfully."
    )


def ensure_required_properties(client):

    collection = client.collections.use(
        settings.weaviate_collection
    )

    config = collection.config.get()

    existing_properties = {
        prop.name for prop in config.properties
    }

    required_properties = {
        "document_id": DataType.TEXT,
        "file_hash": DataType.TEXT,
    }

    for property_name, data_type in required_properties.items():

        if property_name not in existing_properties:

            collection.config.add_property(
                Property(
                    name=property_name,
                    data_type=data_type,
                )
            )

            print(
                f"Added '{property_name}' property."
            )

        else:
            print(
                f"'{property_name}' property already exists."
            )