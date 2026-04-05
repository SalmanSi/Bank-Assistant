"""
Security Pipeline module for the NUST Bank Assistant.

Provides a multi-layered guardrail system to prevent prompt injections, 
jailbreak attempts, off-topic requests, and sensitive data leakage (PII/Card Numbers).
Utilizes RegEx checks alongside ML-based llm-guard models.
"""
from __future__ import annotations

import logging
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Final

try:
    from llm_guard import scan_output as _lg_scan_output
    from llm_guard import scan_prompt as _lg_scan_prompt
    _LG_CORE_AVAILABLE = True
except ImportError:
    _lg_scan_output = None  # type: ignore[assignment]
    _lg_scan_prompt = None  # type: ignore[assignment]
    _LG_CORE_AVAILABLE = False

try:
    from llm_guard.input_scanners import BanTopics as _InputBanTopics
    from llm_guard.input_scanners import Gibberish as _Gibberish
    from llm_guard.input_scanners import InvisibleText as _InvisibleText
    from llm_guard.input_scanners import PromptInjection as _PromptInjection
    from llm_guard.input_scanners import TokenLimit as _TokenLimit
    from llm_guard.input_scanners import Toxicity as _InputToxicity
    from llm_guard.input_scanners.prompt_injection import MatchType as _MatchType
    _LG_INPUT_AVAILABLE = True
except ImportError:
    _LG_INPUT_AVAILABLE = False

try:
    from llm_guard.output_scanners import BanTopics as _OutputBanTopics
    _LG_OUTPUT_AVAILABLE = True
except ImportError:
    _LG_OUTPUT_AVAILABLE = False

logger = logging.getLogger("RAG")

BLOCK_RESPONSE: Final[str] = (
    "I'm sorry, I can only answer questions about NUST Bank products and services."
)

MAX_INPUT_CHARS: Final[int] = 2_000
MAX_INPUT_TOKENS: Final[int] = 500

BANNED_INPUT_TOPICS: Final[list[str]] = [
    "violence", "politics", "religion", "drugs",
    "hacking", "gambling", "adult content", "weapons",
]

# "adult content" is intentionally excluded from output topics. The zero-shot
# classifier (roberta-base-zeroshot-v2.0-c) scores common financial terms like
# "savings", "deposit" and "balance" above 0.60 on this label, which would
# block legitimate banking responses. The other four topics score clearly on
# genuinely harmful output and do not have this overlap problem.
BANNED_OUTPUT_TOPICS: Final[list[str]] = [
    "violence", "politics", "drugs", "hacking",
]

INJECTION_THRESHOLD: Final[float] = 0.5
TOXICITY_THRESHOLD: Final[float]  = 0.5

# The input threshold is higher than the output threshold because short banking
# queries are more ambiguous and the classifier is more likely to misfire on them.
# Long-form LLM responses score more distinctly, so 0.60 is sufficient there.
INPUT_TOPICS_THRESHOLD: Final[float]  = 0.75
OUTPUT_TOPICS_THRESHOLD: Final[float] = 0.60

_JAILBREAK_PATTERNS: Final[list[str]] = [
    # Instruction override
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+instructions?",
    r"forget\s+(all\s+)?(your\s+)?(rules?|instructions?|guidelines?|training)",
    r"forget\s+every(thing)?\s+(you\s+)?(were\s+)?(told|given|taught)",
    r"override\s+(your\s+)?(instructions?|rules?|guidelines?|programming)",
    r"new\s+core\s+directive",
    r"from\s+now\s+on\s+(you\s+are|act|behave|respond)",
    # Persona hijack
    r"\bact\s+as\b(?!\s+a\s+bank)",
    r"\byou\s+are\s+now\b",
    r"\brole\s*-?\s*play\s+as\b",
    r"\bsimulate\b",
    r"\bpretend\s+(you\s+are|to\s+be)\b",
    # Named jailbreak modes
    r"\bDAN\b",
    r"\bdeveloper\s+mode\b",
    r"\buncensored\b",
    r"\bjailbreak\b",
    r"\bgod\s+mode\b",
    r"\bno[\s-]restrictions?\b",
    r"\bno[\s-]filter\b",
    # LLM prompt template tokens used as injection vectors
    r"<\s*/?\s*(s|system|im_start|im_end|SYS|INST|user|assistant)\s*>",
    r"<\|.*?\|>",
    r"\[INST\]",
    r"<<SYS>>",
    r"###\s*SYSTEM",
    # Prompt disclosure: two patterns needed because the verb and target are
    # sometimes separated by several words ("display the contents of your context window")
    r"(show|print|reveal|output|repeat|display|write\s+out|read(\s+me)?)\s+"
    r"(me\s+)?(your\s+|the\s+)?(full\s+|hidden\s+|real\s+|actual\s+|complete\s+)?"
    r"(system\s+prompt|instructions?|prompt|context(\s+window)?|directives?|rules?|guidelines?)",
    r"(show|print|reveal|output|repeat|display|write\s+out|read(\s+me)?).*?"
    r"(your\s+)?(system\s+prompt|context\s+window|instructions?|directives?|rules?|guidelines?)",
    r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions?|rules?|guidelines?)",
    # Hypothetical/fictional framing used to bypass safety rules
    r"hypothetically\s+(speaking\s+)?(if|assume|suppose)",
    r"in\s+a\s+(fictional|hypothetical)\s+(world|scenario|context|universe)",
    r"as\s+a\s+thought\s+experiment",
    r"if\s+you\s+had\s+(no|zero|absolutely\s+no)\s+(rules?|restrictions?|guidelines?|constraints?|limits?)",
    # Context separator abuse and encoding tricks
    r"[-_]{5,}",
    r"={5,}",
    r"\bbase64\b",
    r"\brot\s*13\b",
]

_OFF_TOPIC_PATTERNS: Final[list[str]] = [
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|story|song|essay|haiku|limerick|ballad|narrative|script|screenplay)\b",
    r"\bcompose\s+(a\s+)?(song|poem|melody|tune|haiku|essay)\b",
    r"\brecipe\b",
    r"\bweather\b",
    r"\bsports?\b",
    r"\bpolitics?\b",
    r"\bvideo\s+game(s)?\b",
    r"\bcelebrit(y|ies)\b",
    r"\bmovie(s)?\b",
    r"\bmusic\b",
    r"\bessay\s+on\b",
    r"\bshort\s+story\b",
]

# Patterns that should never appear in LLM output. These catch system prompt
# leakage, PII, and credentials that the model should never reproduce.
_SENSITIVE_OUTPUT_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"OFFICIAL NUST BANK KNOWLEDGE (START|END)", "system prompt marker"),
    (r"SYSTEM_PROMPT_TEMPLATE",                   "internal variable name"),
    (r"\b\d{13,19}\b",                            "potential card/account number"),
    (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b", "IBAN"),
    (r"password\s*[:=]\s*\S+",                    "password field"),
    (r"api[_\s-]?key\s*[:=]\s*\S+",               "API key"),
    (r"secret\s*[:=]\s*\S+",                      "secret field"),
]

_COMPILED_JAILBREAK: Final[list[re.Pattern]] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in _JAILBREAK_PATTERNS
]
_COMPILED_OFF_TOPIC: Final[list[re.Pattern]] = [
    re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS
]
_COMPILED_SENSITIVE: Final[list[tuple[re.Pattern, str]]] = [
    (re.compile(p, re.IGNORECASE), label)
    for p, label in _SENSITIVE_OUTPUT_PATTERNS
]


@dataclass
class GuardResult:
    """Returned by every pipeline check. Inspect .allowed first."""
    allowed: bool
    layer: str = ""
    reason: str = ""
    risk_scores: dict = field(default_factory=dict)
    safe_response: str = BLOCK_RESPONSE


def _build_input_scanners() -> list:
    if not (_LG_CORE_AVAILABLE and _LG_INPUT_AVAILABLE):
        return []

    loaded: list = [
        _TokenLimit(limit=MAX_INPUT_TOKENS),
        _InvisibleText(),
    ]
    for name, cls, kwargs in [
        ("Gibberish",       _Gibberish,       {}),
        ("Toxicity",        _InputToxicity,   {"threshold": TOXICITY_THRESHOLD}),
        ("BanTopics",       _InputBanTopics,  {"topics": BANNED_INPUT_TOPICS, "threshold": INPUT_TOPICS_THRESHOLD}),
        ("PromptInjection", _PromptInjection, {"threshold": INJECTION_THRESHOLD, "match_type": _MatchType.FULL}),
    ]:
        try:
            loaded.append(cls(**kwargs))
            logger.info("  [llm-guard] input scanner loaded: %s", name)
        except Exception:
            logger.error("  [llm-guard] input scanner '%s' failed to load:\n%s", name, traceback.format_exc())
    return loaded


def _build_output_scanners() -> list:

    if not (_LG_CORE_AVAILABLE and _LG_OUTPUT_AVAILABLE):
        return []

    loaded: list = []
    for name, cls, kwargs in [
        ("BanTopics", _OutputBanTopics, {"topics": BANNED_OUTPUT_TOPICS, "threshold": OUTPUT_TOPICS_THRESHOLD}),
    ]:
        try:
            loaded.append(cls(**kwargs))
            logger.info("  [llm-guard] output scanner loaded: %s", name)
        except Exception:
            logger.error("  [llm-guard] output scanner '%s' failed to load:\n%s", name, traceback.format_exc())
    return loaded


class GuardPipeline:
    """
    Security pipeline for the NUST Bank assistant.

    Three-layer architecture:
        Layer 0 - sanity checks (type, length)
        Layer 1 - regex patterns for jailbreaks, off-topic requests, and sensitive output
        Layer 2 - ML-based scanners via llm-guard (PromptInjection, Toxicity, BanTopics, etc.)

    Instantiate once at startup using get_pipeline().
    Set LLMGUARD_DISABLE=1 to skip ML layer (useful in tests).
    """

    def __init__(self) -> None:
        if os.environ.get("LLMGUARD_DISABLE", "").strip() == "1":
            logger.info("[llm-guard] disabled via LLMGUARD_DISABLE=1")
            self._input_scanners: list  = []
            self._output_scanners: list = []
            self._ml_active = False
            return

        if not _LG_CORE_AVAILABLE:
            logger.warning("[llm-guard] package not found, ML scanning disabled. Run: pip install llm-guard")
            self._input_scanners  = []
            self._output_scanners = []
            self._ml_active = False
            return

        self._input_scanners  = _build_input_scanners()
        self._output_scanners = _build_output_scanners()
        self._ml_active = bool(self._input_scanners or self._output_scanners)
        logger.info("[llm-guard] ready: %d input scanner(s), %d output scanner(s)",
                    len(self._input_scanners), len(self._output_scanners))

    def check_input(self, user_message: str) -> GuardResult:
        if not isinstance(user_message, str):
            return self._block("layer0", "non-string input")
        text = user_message.strip()
        if not text:
            return self._block("layer0", "empty input")
        if len(text) > MAX_INPUT_CHARS:
            return self._block("layer0", "input exceeds max length")

        for pattern in _COMPILED_JAILBREAK:
            m = pattern.search(text)
            if m:
                logger.warning("[guard] jailbreak matched: %r -> %r", pattern.pattern, m.group())
                return self._block("layer1", f"jailbreak: {pattern.pattern}")

        for pattern in _COMPILED_OFF_TOPIC:
            m = pattern.search(text)
            if m:
                logger.info("[guard] off-topic matched: %r -> %r", pattern.pattern, m.group())
                return self._block("layer1", f"off-topic: {pattern.pattern}")

        if self._ml_active and self._input_scanners:
            _, valid_map, score_map = _lg_scan_prompt(self._input_scanners, text, fail_fast=True)
            failed = {k for k, v in valid_map.items() if not v}
            if failed:
                logger.warning("[guard] ML input blocked: %s scores=%s", failed, score_map)
                return GuardResult(allowed=False, layer="layer2",
                                   reason=f"llm-guard failed: {failed}",
                                   risk_scores=score_map, safe_response=BLOCK_RESPONSE)

        return GuardResult(allowed=True)

    def check_output(self, prompt: str, llm_response: str) -> GuardResult:
        if not isinstance(llm_response, str):
            return self._block("layer0_out", "non-string LLM output")

        for pattern, label in _COMPILED_SENSITIVE:
            if pattern.search(llm_response):
                logger.warning("[guard] sensitive data in output: %s", label)
                return self._block("layer1_out", f"sensitive data: {label}")

        if self._ml_active and self._output_scanners:
            _, valid_map, score_map = _lg_scan_output(
                self._output_scanners, prompt, llm_response, fail_fast=True)
            failed = {k for k, v in valid_map.items() if not v}
            if failed:
                logger.warning("[guard] ML output blocked: %s scores=%s", failed, score_map)
                return GuardResult(allowed=False, layer="layer2_out",
                                   reason=f"llm-guard output failed: {failed}",
                                   risk_scores=score_map, safe_response=BLOCK_RESPONSE)

        return GuardResult(allowed=True)

    @staticmethod
    def _block(layer: str, reason: str) -> GuardResult:
        return GuardResult(allowed=False, layer=layer, reason=reason, safe_response=BLOCK_RESPONSE)


_pipeline: GuardPipeline | None = None


def get_pipeline() -> GuardPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GuardPipeline()
    return _pipeline


def check_input(user_message: str) -> GuardResult:
    return get_pipeline().check_input(user_message)


def check_output(prompt: str, llm_response: str) -> GuardResult:
    return get_pipeline().check_output(prompt, llm_response)