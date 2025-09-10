from typing import Literal

from cachetools import TTLCache, cached
from langchain_core.embeddings import Embeddings
from langchain_milvus.vectorstores.milvus import Milvus

VectorProvider = Literal["milvus"]


def provider_factotry(maxsize: int = 10, ttl: float = 360):
    cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def cache_key(
        provider, name, database, host="localhost", port=19530, *args, **kwargs
    ) -> str:
        return f"{provider}_{name}_{database}_{host}_{port}"

    @cached(cache, key=cache_key)
    def factory(
        provider: VectorProvider,
        name: str,
        database: str,
        embeddings: Embeddings,
        description: str | None = "",
        host: str | None = "localhost",
        port: int | None = 19530,
    ):
        match provider:
            case "milvus":
                return Milvus(
                    embedding_function=embeddings,
                    collection_name=name,
                    collection_description=description,
                    connection_args={
                        "uri": f"http://{host}:{port}",
                        "host": host,
                        "port": port,
                        "db_name": database,
                    },
                    enable_dynamic_field=True,
                    auto_id=True,
                    search_params={"metric_type": "L2", "params": {"ef": 250}},
                )
            case _:
                ...

    return factory
