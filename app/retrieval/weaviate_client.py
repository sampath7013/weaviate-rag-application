import weaviate
from weaviate.classes.config import Configure, Property, DataType

from app.config import settings


def get_weaviate_client():
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )


def create_collection(client):
    collection_name = settings.weaviate_collection

    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
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


def ensure_document_id_property(client):
    collection = client.collections.use(
        settings.weaviate_collection
    )

    config = collection.config.get()

    property_names = {
        prop.name for prop in config.properties
    }

    if "document_id" not in property_names:
        collection.config.add_property(
            Property(
                name="document_id",
                data_type=DataType.TEXT,
            )
        )

        print("Added 'document_id' property.")

    else:
        print("'document_id' property already exists.")