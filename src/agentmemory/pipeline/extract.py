"""Extract Pipeline - LLM-based fact/entity/relation extraction.

References:
    - ARCHITECTURE.md §10.2 (lines 1516-1533)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import MemoryItem
    from ..core.llm import LLMProvider

_DEFAULT_EXTRACT_PROMPT = """You are a fact extraction system. Given the following content, extract key facts, entities, and relations as a JSON list.

Output format:
{
  "facts": [
    {
      "content": "fact text",
      "entities": ["entity1", "entity2"],
      "importance": 0.5,
      "type": "semantic"
    }
  ]
}

Rules:
- Extract only verifiable factual statements
- List all entities mentioned (people, places, organizations, etc.)
- Set importance from 0.0 to 1.0 based on likely usefulness
- Types: semantic, procedural, reflective, user

Content to analyze:
{content}

Output JSON:"""


class ExtractPipeline:
    """Extract facts/entities/relations via LLM.

    Input: messages or raw content
    Output: list of MemoryItem
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompt_template: str = _DEFAULT_EXTRACT_PROMPT,
        dedupe_threshold: float = 0.88,
        max_retries: int = 2,
        tenant_id: str = "default",
        namespace: str = "default",
    ):
        """Initialize ExtractPipeline.

        Args:
            llm: LLMProvider instance for extraction
            prompt_template: Prompt template with {content} placeholder
            dedupe_threshold: Vector similarity threshold for fact dedup (0.88)
            max_retries: Max JSON parse retry attempts
            tenant_id: Tenant identifier
            namespace: Namespace identifier
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.dedupe_threshold = dedupe_threshold
        self.max_retries = max_retries
        self.tenant_id = tenant_id
        self.namespace = namespace

    async def extract(self, messages: list[dict], source: str = "inference") -> list[MemoryItem]:
        """Extract facts from a message list.

        Args:
            messages: List of message dicts with 'role' and 'content'
            source: Source identifier (user/system/inference/reflection)

        Returns:
            List of extracted MemoryItem
        """
        # Build content from messages
        if not messages:
            return []

        if isinstance(messages[0], dict) and "content" in messages[0]:
            content = "\n".join(
                f"{m.get('role', 'user')}: {m['content']}" for m in messages if m.get("content")
            )
        else:
            content = str(messages[0]) if messages else ""

        return await self.extract_one(content, source=source)

    async def extract_one(self, content: str, source: str = "inference") -> list[MemoryItem]:
        """Extract facts from a single content string.

        Args:
            content: Raw text content
            source: Source identifier

        Returns:
            List of extracted MemoryItem
        """
        from ..core.types import MemoryItem, MemoryLayer, MemoryType

        prompt = self.prompt_template.format(content=content)

        # Try LLM extraction with retry
        for attempt in range(self.max_retries):
            try:
                response = await self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Low temperature for structured extraction
                    max_tokens=2048,
                )

                # Parse JSON from response
                text = response.content.strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                text = text.strip()

                parsed = json.loads(text)
                facts = parsed.get("facts", [])

                items = []
                for fact_data in facts:
                    item = MemoryItem(
                        content=fact_data.get("content", ""),
                        type=MemoryType(fact_data.get("type", "semantic")),
                        layer=MemoryLayer.L2_GRAPH,  # Extraction goes to L2
                        importance=fact_data.get("importance", 0.5),
                        entities=fact_data.get("entities", []),
                        source=source,
                        tenant_id=self.tenant_id,
                        namespace=self.namespace,
                        metadata={},
                    )
                    items.append(item)

                return items

            except json.JSONDecodeError:
                if attempt < self.max_retries - 1:
                    continue
                # Fall through to regex-based fallback
                return self._regex_fallback(content, source)

        return []

    def _regex_fallback(self, content: str, source: str) -> list[MemoryItem]:
        """Fallback regex-based extraction when LLM fails.

        Args:
            content: Raw text content
            source: Source identifier

        Returns:
            List of MemoryItem with basic regex extraction
        """
        import re

        from ..core.types import MemoryItem, MemoryLayer, MemoryType

        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        items = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence) < 10:
                continue

            # Extract potential entities (capitalized words)
            entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)

            item = MemoryItem(
                content=sentence,
                type=MemoryType.SEMANTIC,
                layer=MemoryLayer.L2_GRAPH,
                importance=0.3,  # Lower importance for regex fallback
                entities=entities[:5],  # Limit entities
                source=source,
                tenant_id=self.tenant_id,
                namespace=self.namespace,
                metadata={"extraction_method": "regex_fallback"},
            )
            items.append(item)

        return items
