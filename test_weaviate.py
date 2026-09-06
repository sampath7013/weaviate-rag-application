from app.retrieval.weaviate_client import (
    get_weaviate_client,
    create_collection,
    ensure_required_properties,
)


client = get_weaviate_client()

try:

    print(
        "Connected:",
        client.is_connected(),
    )

    print(
        "Ready:",
        client.is_ready(),
    )

    create_collection(client)

    ensure_required_properties(client)

finally:
    client.close()