"""
LLM-based Fact Extractor
Implements FactExtractor Protocol
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class ExtractedFact:
    """Extracted fact representation"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str | None = None


@dataclass
class ExtractionResult:
    """Fact extraction result"""
    facts: list[ExtractedFact]
    summary: str | None = None
    metadata: dict[str, Any] | None = None


class FactExtractor:
    """Protocol for fact extraction"""

    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """Extract facts from text"""
        raise NotImplementedError

    async def aextract(self, text: str, **kwargs) -> ExtractionResult:
        """Async fact extraction"""
        raise NotImplementedError


class LLMFactExtractor(FactExtractor):
    """
    LLM-based fact extractor using structured output.
    Extracts subject-predicate-object triplets from text.
    """

    def __init__(
        self,
        llm_provider: Any | None = None,
        model: str | None = None,
        prompt_template: str | None = None,
        **kwargs
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.prompt_template = prompt_template or self._default_prompt()
        self.kwargs = kwargs

    def _default_prompt(self) -> str:
        """Default prompt for fact extraction"""
        return """Extract factual triplets (subject, predicate, object) from the following text.
Return a JSON array of triplets with fields: subject, predicate, object, confidence (0-1).

Text: {text}

Output only the JSON array, no explanation."""

    def _parse_response(self, content: str) -> list[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects"""
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        facts = []
        for item in data:
            if isinstance(item, dict):
                facts.append(ExtractedFact(
                    subject=item.get("subject", ""),
                    predicate=item.get("predicate", ""),
                    object=item.get("object", ""),
                    confidence=item.get("confidence", 0.5),
                    source=None
                ))

        return facts

    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """Extract facts from text using LLM"""
        if not self.llm_provider:
            raise ValueError("LLM provider required for fact extraction")

        prompt = self.prompt_template.format(text=text)

        try:
            response = self.llm_provider.complete(prompt, model=self.model, **self.kwargs)
            content = response.content if hasattr(response, 'content') else str(response)
            facts = self._parse_response(content)

            return ExtractionResult(
                facts=facts,
                summary=f"Extracted {len(facts)} facts" if facts else None,
                metadata={"model": self.model} if self.model else None
            )
        except Exception as e:
            return ExtractionResult(
                facts=[],
                summary=f"Extraction failed: {str(e)}",
                metadata={"error": str(e)}
            )

    async def aextract(self, text: str, **kwargs) -> ExtractionResult:
        """Async fact extraction"""
        if not self.llm_provider:
            raise ValueError("LLM provider required for fact extraction")

        prompt = self.prompt_template.format(text=text)

        try:
            response = await self.llm_provider.acomplete(prompt, model=self.model, **self.kwargs)
            content = response.content if hasattr(response, 'content') else str(response)
            facts = self._parse_response(content)

            return ExtractionResult(
                facts=facts,
                summary=f"Extracted {len(facts)} facts" if facts else None,
                metadata={"model": self.model} if self.model else None
            )
        except Exception as e:
            return ExtractionResult(
                facts=[],
                summary=f"Extraction failed: {str(e)}",
                metadata={"error": str(e)}
            )
