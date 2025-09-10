from typing import Literal

from cachetools import TTLCache, cached
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

EmbeddingProvider = Literal["openai"]


def provider_factory(maxsize: int = 10, ttl: float = 360):
    cache = TTLCache(maxsize=maxsize, ttl=ttl)

    @cached(
        cache,
        key=lambda provider,
        model,
        base_url,
        *args,
        **kwargs: f"{provider}_{model}_{base_url}",
    )
    def factory(
        provider: EmbeddingProvider, model: str, base_url: str, api_key: str
    ) -> Embeddings:
        match provider:
            case "openai":
                embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_base=base_url,
                    openai_api_key=api_key,
                    check_embedding_ctx_length=False,
                )
            case _:
                ...

        return embeddings

    return factory
