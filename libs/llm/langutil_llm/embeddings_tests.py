import os
from langutil_llm.embeddings import emb_factory

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:8750/v1")


def test_factory():
    factory = emb_factory()

    kwds = {
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": MODEL_BASE_URL,
        "api_key": "apikey",
    }

    emb1 = factory(**kwds)
    emb2 = factory(**kwds)

    assert emb1 == emb2


def test_factory2():
    factory = emb_factory()

    kwds = {
        "provider": "openai",
        "base_url": MODEL_BASE_URL,
        "api_key": "apikey",
    }

    emb1 = factory(**kwds, model="gpt-4o")
    emb2 = factory(**kwds, model="gpt-3.5-turbo")

    assert emb1 != emb2
