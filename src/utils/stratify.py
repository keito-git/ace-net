"""
Context-deficiency stratification for sub-group evaluation.

Defines four strata that capture hard cases where explicit emotion signals
are absent or misleading, the central challenge for ACE-Net.

Strata:
  1. no_explicit_emotion : no explicit emotion word in text (NRC-style word list)
  2. short_text          : fewer than 10 whitespace tokens
  3. irony_sarcasm       : heuristic sarcasm/irony surface markers
  4. implicit_presup     : implicit presupposition cues (negation + positive
                           content, counterfactual, wish constructions)

These are heuristic approximations used for diagnostic sub-group analysis,
not ground-truth filtering. A sample may belong to multiple strata.
"""
import re
from typing import Dict, List

# Subset of the NRC Emotion Lexicon plus common English emotion terms.
# Source: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
_EXPLICIT_EMOTION_WORDS: frozenset[str] = frozenset([
    "happy", "happiness", "joy", "joyful", "glad", "pleased", "delighted",
    "thrilled", "ecstatic", "elated", "content", "satisfied", "cheerful",
    "excited", "excited", "euphoric", "blissful", "enthusiastic", "grateful",
    "sad", "sadness", "unhappy", "depressed", "miserable", "sorrow", "sorrowful",
    "grief", "grieving", "heartbroken", "melancholy", "gloomy", "dejected",
    "disappointed", "disappointment", "despair", "hopeless", "regret",
    "angry", "anger", "furious", "rage", "irritated", "annoyed", "frustrated",
    "outraged", "resentful", "enraged", "mad", "irate", "hostile",
    "fear", "fearful", "afraid", "scared", "terrified", "anxious", "anxiety",
    "nervous", "worried", "apprehensive", "dread", "horror", "horrified",
    "panicked", "panic",
    "disgust", "disgusted", "revolted", "repulsed", "nauseated", "loathe",
    "surprised", "surprise", "astonished", "amazed", "shocked", "startled",
    "stunned", "bewildered",
    "love", "adore", "trust", "proud", "pride", "admire", "admiration",
    "hopeful", "excited", "anticipate", "eager", "longing",
    "embarrassed", "embarrassment", "ashamed", "shame", "guilty", "guilt",
    "jealous", "jealousy", "envy", "nostalgic", "nostalgic", "confused",
    "bored", "boredom", "lonely", "loneliness", "amused", "amusement",
    "contempt", "pity", "empathy", "compassion",
])

# Heuristic sarcasm/irony surface markers (some false positives acceptable).
_SARCASM_PATTERNS: list[str] = [
    r"\b/s\b",
    r"(?i)\byeah\s+right\b",
    r"(?i)\boh\s+(great|wonderful|fantastic|brilliant|sure)\b",
    r"(?i)\b(totally|absolutely|definitely|certainly)\s+not\b",
    r"(?i)\bwhat\s+a\s+(great|wonderful|lovely|fantastic)\b",
    r"(?i)\bi\s+(love|enjoy)\s+how\b",
    r"(?i)\bso\s+helpful\b",
    r"(?i)\bthanks\s+(a\s+lot|so\s+much)\b.*[!]{2,}",
]

# Implicit presupposition cues: counterfactuals, unfulfilled wishes,
# negation of a positive state.
_IMPLICIT_PRESUP_PATTERNS: list[str] = [
    r"(?i)\bif\s+only\b",
    r"(?i)\bi\s+wish\b",
    r"(?i)\bcould\s+have\s+been\b",
    r"(?i)\bshould\s+have\b",
    r"(?i)\bwould\s+have\b",
    r"(?i)\bi\s+can'?t\s+believe\b",
    r"(?i)\bnot\s+(happy|okay|ok|fine|good|great|wonderful|satisfied)\b",
    r"(?i)\bno\s+longer\b",
    r"(?i)\bnever\s+(thought|expected|imagined|believed)\b",
    r"(?i)\bused\s+to\b",
]

_COMPILED_SARCASM = [re.compile(p) for p in _SARCASM_PATTERNS]
_COMPILED_IMPLICIT = [re.compile(p) for p in _IMPLICIT_PRESUP_PATTERNS]
_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _has_explicit_emotion(text: str) -> bool:
    """Return True if text contains at least one explicit emotion word."""
    tokens = {m.group().lower() for m in _TOKEN_PATTERN.finditer(text)}
    return bool(tokens & _EXPLICIT_EMOTION_WORDS)


def _is_short_text(text: str, threshold: int = 10) -> bool:
    """Return True if text has fewer than `threshold` whitespace tokens."""
    return len(text.split()) < threshold


def _is_irony_sarcasm(text: str) -> bool:
    """Return True if any sarcasm/irony heuristic fires."""
    return any(p.search(text) for p in _COMPILED_SARCASM)


def _has_implicit_presup(text: str) -> bool:
    """Return True if any implicit presupposition cue is present."""
    return any(p.search(text) for p in _COMPILED_IMPLICIT)


def compute_strata_flags(texts: List[str]) -> Dict[str, List[bool]]:
    """
    Compute the four context-deficiency stratum membership flags per text.

    Returns a dict with boolean lists for keys:
      'no_explicit_emotion', 'short_text', 'irony_sarcasm', 'implicit_presup'.
    """
    return {
        "no_explicit_emotion": [not _has_explicit_emotion(t) for t in texts],
        "short_text": [_is_short_text(t) for t in texts],
        "irony_sarcasm": [_is_irony_sarcasm(t) for t in texts],
        "implicit_presup": [_has_implicit_presup(t) for t in texts],
    }


def describe_strata(texts: List[str]) -> Dict[str, int]:
    """Return the count of texts in each stratum (for dataset characterization)."""
    flags = compute_strata_flags(texts)
    return {name: sum(mask) for name, mask in flags.items()}
