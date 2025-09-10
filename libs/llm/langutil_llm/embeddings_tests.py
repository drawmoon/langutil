from langutil_llm.embeddings import emb_factory


def test_factory():
    factory = emb_factory()

    kwds = {
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": "http://localhost:8750/v1",
        "api_key": "apikey",
    }

    emb1 = factory(**kwds)
    emb2 = factory(**kwds)

    assert emb1 == emb2


def test_factory2():
    factory = emb_factory()

    kwds = {
        "provider": "openai",
        "base_url": "http://localhost:8750/v1",
        "api_key": "apikey",
    }

    emb1 = factory(**kwds, model="gpt-4o")
    emb2 = factory(**kwds, model="gpt-3.5-turbo")

    assert emb1 != emb2
