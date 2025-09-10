from typing import Any, Literal, Protocol, overload, runtime_checkable

from cachetools import TTLCache, cached
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

EmbeddingProvider = Literal["openai"]


@runtime_checkable
class EmbFactory(Protocol):
    def __call__(
        self,
        provider: EmbeddingProvider,
        model: str,
        base_url: str,
        api_key: str,
        **kwargs: Any,
    ) -> Embeddings: ...


@overload
def emb_factory() -> EmbFactory: ...
@overload
def emb_factory(maxsize: int, ttl: float) -> EmbFactory: ...
@overload
def emb_factory(cache: Any) -> EmbFactory: ...
def emb_factory(
    maxsize: int | None = 10, ttl: float | None = 360, cache: Any | None = None
) -> EmbFactory:
    cache = cache if cache is not None else TTLCache(maxsize=maxsize, ttl=ttl)

    @cached(
        cache,
        key=lambda provider,
        model,
        base_url,
        *args,
        **kwargs: f"{provider}_{model}_{base_url}",
    )
    def factory(
        provider: EmbeddingProvider,
        model: str,
        base_url: str,
        api_key: str,
        **kwargs: Any,
    ) -> Embeddings:
        match provider:
            case "openai":
                embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_base=base_url,
                    openai_api_key=api_key,
                    check_embedding_ctx_length=False,
                    **kwargs,
                )
            case _:
                ...

        return embeddings

    return factory
