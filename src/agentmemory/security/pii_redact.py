"""PII Redaction - Personal Identifiable Information redaction component.

References:
    - ARCHITECTURE.md §7.4 (data privacy)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Pattern

from ..observability.events import Event, EventBus, EventType


@dataclass
class PIIRule:
    """A PII detection rule with regex pattern and replacement."""
    name: str
    pattern: str | Pattern[str]
    replacement: str = "[REDACTED]"
    flags: int = re.IGNORECASE

    def __post_init__(self):
        if isinstance(self.pattern, str):
            self._pattern = re.compile(self.pattern, self.flags)
        else:
            self._pattern = self.pattern

    @property
    def compiled(self) -> Pattern[str]:
        return self._pattern


# Default PII rules for common personal information
DEFAULT_PII_RULES: list[PIIRule] = [
    # Email addresses
    PIIRule(
        name="email",
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        replacement="[EMAIL]"
    ),
    # Chinese mobile phone numbers
    PIIRule(
        name="phone_cn",
        pattern=r'1[3-9]\d{9}',
        replacement="[PHONE]"
    ),
    # US phone numbers
    PIIRule(
        name="phone_us",
        pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        replacement="[PHONE]"
    ),
    # Chinese ID card numbers (18 digits)
    PIIRule(
        name="id_card_cn",
        pattern=r'\b[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b',
        replacement="[ID_CARD]"
    ),
    # Chinese resident ID card (15 digits, older format)
    PIIRule(
        name="id_card_cn_15",
        pattern=r'\b[1-9]\d{5}\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}\b',
        replacement="[ID_CARD]"
    ),
    # Bank card numbers (16-19 digits)
    PIIRule(
        name="bank_card",
        pattern=r'\b\d{16,19}\b',
        replacement="[BANK_CARD]"
    ),
    # IP addresses
    PIIRule(
        name="ip_address",
        pattern=r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        replacement="[IP]"
    ),
    # Chinese address patterns (simplified)
    PIIRule(
        name="address_cn",
        pattern=r'(?:省|市|区|县|路|街|巷|号)\S{0,30}(?:省|市|区|县|路|街|巷|号)',
        replacement="[ADDRESS]"
    ),
    # Passport numbers (generic pattern)
    PIIRule(
        name="passport",
        pattern=r'\b[A-Z]{1,2}\d{6,9}\b',
        replacement="[PASSPORT]"
    ),
    # Social security numbers (US format)
    PIIRule(
        name="ssn",
        pattern=r'\b\d{3}-\d{2}-\d{4}\b',
        replacement="[SSN]"
    ),
]


class PIIRedactor:
    """PII (Personally Identifiable Information) redactor.

    Detects and redacts common PII types using regex patterns.
    Supports custom rules and structured data redaction.

    Example:
        >>> redactor = PIIRedactor()
        >>> redactor.redact("Contact me at john@example.com")
        'Contact me at [EMAIL]'
        >>> redactor.redact_struct({"email": "test@test.com"})
        {'email': '[EMAIL]'}
    """

    def __init__(
        self,
        rules: list[PIIRule] | None = None,
        event_bus: EventBus | None = None,
        detect_only: bool = False
    ):
        """Initialize PIIRedactor.

        Args:
            rules: Custom PII rules. Uses defaults if None.
            event_bus: Optional EventBus for logging detections.
            detect_only: If True, only detect (don't redact).
        """
        self._rules = rules if rules is not None else DEFAULT_PII_RULES.copy()
        self._event_bus = event_bus
        self._detect_only = detect_only
        self._stats = {"total_detections": 0, "by_type": {}}

    @property
    def rules(self) -> list[PIIRule]:
        """Get list of active PII rules."""
        return self._rules

    def add_rule(self, rule: PIIRule) -> None:
        """Add a custom PII detection rule.

        Args:
            rule: PIIRule to add.
        """
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Args:
            name: Name of rule to remove.

        Returns:
            True if rule was found and removed.
        """
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def redact(self, text: str) -> str:
        """Redact PII from text.

        Args:
            text: Input text to redact.

        Returns:
            Text with PII replaced by placeholders.
        """
        if not text:
            return text

        redacted = text

        for rule in self._rules:
            matches = list(rule.compiled.finditer(text))
            if matches:
                # Track detection stats
                self._stats["total_detections"] += len(matches)
                self._stats["by_type"][rule.name] = (
                    self._stats["by_type"].get(rule.name, 0) + len(matches)
                )

                # Emit event if configured
                if self._event_bus is not None:
                    for match in matches:
                        event = Event(
                            type=EventType.PII_DETECTED,
                            payload={
                                "rule": rule.name,
                                "matched": match.group(),
                                "position": match.span(),
                            }
                        )
                        # Fire-and-forget async emit
                        try:
                            import asyncio
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(self._event_bus.emit(event))
                            else:
                                loop.run_until_complete(self._event_bus.emit(event))
                        except Exception:
                            pass  # Don't let event failures affect redaction

                if not self._detect_only:
                    redacted = rule.compiled.sub(rule.replacement, redacted)

        return redacted

    def detect(self, text: str) -> list[dict[str, Any]]:
        """Detect PII without redacting.

        Args:
            text: Input text to scan.

        Returns:
            List of detection results with rule name, match, and position.
        """
        results = []
        for rule in self._rules:
            for match in rule.compiled.finditer(text):
                results.append({
                    "rule": rule.name,
                    "match": match.group(),
                    "span": match.span(),
                })
        return results

    def redact_struct(self, obj: Any) -> Any:
        """Recursively redact PII from structured data.

        Handles dicts, lists, strings, and other primitives.
        Dicts keys are preserved, only values are redacted.

        Args:
            obj: Structured data (dict, list, str, etc.)

        Returns:
            Same structure with PII redacted.
        """
        if isinstance(obj, str):
            return self.redact(obj)
        elif isinstance(obj, dict):
            return {k: self.redact_struct(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.redact_struct(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.redact_struct(item) for item in obj)
        else:
            return obj

    def detect_struct(self, obj: Any) -> dict[str, list[dict[str, Any]]]:
        """Recursively detect PII in structured data.

        Args:
            obj: Structured data to scan.

        Returns:
            Dict mapping field paths to detection results.
        """
        results = {}

        def scan(value: Any, path: str = "") -> None:
            if isinstance(value, str):
                detections = self.detect(value)
                if detections:
                    results[path] = detections
            elif isinstance(value, dict):
                for k, v in value.items():
                    new_path = f"{path}.{k}" if path else k
                    scan(v, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    scan(item, f"{path}[{i}]")

        scan(obj)
        return results

    def stats(self) -> dict[str, Any]:
        """Get redaction statistics.

        Returns:
            Dict with total detections and per-rule counts.
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self._stats = {"total_detections": 0, "by_type": {}}
