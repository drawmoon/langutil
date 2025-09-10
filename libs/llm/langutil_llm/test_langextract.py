import os
from langutil.langextract import Document, Example, LangExtractor


MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:8750/v1")


def test_extractor():
    examples = [
        Document(
            text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
            extractions=[
                Example(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"},
                ),
                Example(
                    extraction_class="emotion",
                    extraction_text="But soft!",
                    attributes={"feeling": "gentle awe"},
                ),
                Example(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"},
                ),
            ],
        ),
    ]
    extractor = LangExtractor(
        model="Qwen3-30B-A3B-Instruct-2507",
        base_url=MODEL_BASE_URL,
        api_key="abc",
        examples=examples,
    )
    annotated_documents = extractor.invoke(
        "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"
    )
    assert len(annotated_documents.extractions) > 0
