from typing import Any, Sequence

from langchain_cohere import CohereRerank
from langchain_core.callbacks.base import Callbacks
from langchain_core.documents.base import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import BaseModel


class RankResult(BaseModel): ...


class LlmBaseRerank(BaseDocumentCompressor):
    def rerank(
        self,
        documents: Sequence[str | Document | dict[str, Any]],
        query: str,
        *,
        rank_fields: Sequence[str] | None = None,
        top_n: int | None = -1,
    ) -> list[RankResult]: ...

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]: ...


def provider_factory(maxsize: int = 10, ttl: float = 360):
    def factory(model: str, base_url: str, api_key: str):
        if "rerank" in model:
            CohereRerank(base_url=base_url, model=model, cohere_api_key=api_key)
        else:
            LlmBaseRerank(model=model, base_url=base_url, api_key=api_key)
