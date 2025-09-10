import dataclasses
from textwrap import dedent
from typing import Any, Literal, Optional

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langextract import data as lx_data, extract
from langextract.factory import ModelConfig, create_model
from pydantic import BaseModel, model_validator

EXTRACT_PROMPT = dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

LangAnnotatedDocument = lx_data.AnnotatedDocument
LangExtractResult = LangAnnotatedDocument | list[LangAnnotatedDocument]
LangExampleData = lx_data.ExampleData
LangExtraction = lx_data.Extraction


class Example(BaseModel):
    extraction_class: str
    extraction_text: str
    attributes: Optional[dict[str, str | list[str]]] = None

    def to_lang_extraction(self) -> LangExtraction:
        return LangExtraction(**self.model_dump())


class ExampleData(BaseModel):
    text: Optional[str] = None
    extractions: Optional[list[Example]] = None

    def to_lang_example_data(self) -> LangExampleData:
        return LangExampleData(
            text=self.text,
            extractions=[
                extra.to_lang_extraction() for extra in self.extractions or []
            ],
        )


class Extraction(BaseModel):
    extraction_class: str
    extraction_text: str
    char_interval: Optional[dict[str, Any]] = None
    alignment_status: Optional[str] = None
    extraction_index: Optional[int] = None
    group_index: Optional[int] = None
    description: Optional[str] = None
    attributes: Optional[dict[str, str | list[str]]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, data: Any) -> Any:
        if isinstance(data, LangExtraction):
            return dataclasses.asdict(data)
        return data


class AnnotatedDocument(BaseModel):
    text: Optional[str] = None
    extractions: Optional[list[Extraction]] = None
    document_id: Optional[str] = None
    tokenized_text: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, data: Any) -> Any:
        if isinstance(data, LangAnnotatedDocument):
            return dataclasses.asdict(data)
        return data


class LangExtractor(RunnableSerializable[str, AnnotatedDocument]):
    prompt_template: str = EXTRACT_PROMPT
    examples: list[ExampleData]
    model: str
    base_url: str
    api_key: str
    model_provider: Literal["openai"] = "openai"
    model_args: Optional[dict[str, Any]] = None

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ):
        config = ModelConfig(
            model_id=self.model,
            provider=self.model_provider,
            provider_kwargs={
                **(self.model_args or {}),
                "base_url": self.base_url,
                "api_key": self.api_key,
            },
        )
        model = create_model(config)

        examples = [example.to_lang_example_data() for example in self.examples]

        result: LangExtractResult = extract(
            text_or_documents=input,
            prompt_description=self.prompt_template,
            examples=examples,
            model=model,
        )
        if isinstance(result, list):
            return [AnnotatedDocument.model_validate(doc) for doc in result]
        else:
            return AnnotatedDocument.model_validate(result)
