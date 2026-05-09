import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "birthdy_memory"
VECTOR_SIZE = 768


def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def init_collection() -> None:
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def store_memory(
    session_id: str,
    role: str,
    content: str,
    vector: list[float],
    msg_id: int,
) -> None:
    client = _client()
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=msg_id,
                vector=vector,
                payload={
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                },
            )
        ],
    )


async def search_memory(
    session_id: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[dict]:
    client = _client()
    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    return [
        {"role": r.payload["role"], "content": r.payload["content"], "score": r.score}
        for r in response.points
    ]
