"""Text normalization for Egyptian Arabic, Franco-Arabic, English, and mixed messages."""

from __future__ import annotations

import re

from triage.domain.models import Script
from triage.domain.pii import to_latin_digits

_ARABIC_LETTER = re.compile(r"[ء-يٮ-ۓ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_REPEATED = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"\[[a-z_]+\]|[^\s]+")
_FRANCO_DIGIT_WORD = re.compile(r"^(?=.*[a-z])(?=.*[235679])[a-z0-9]+$")
_PUNCTUATION_EDGES = re.compile(r"^([^\w\[]*)(.*?)([^\w\]]*)$", re.UNICODE)

_ARABIC_UNIFY = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})

_EMOJI_TOKENS = {
    "😡": " غاضب ",
    "🤬": " غاضب ",
    "😠": " غاضب ",
    "😤": " غاضب ",
    "😢": " حزين ",
    "😭": " حزين ",
    "😞": " حزين ",
    "🙏": " رجاء ",
    "👍": " جيد ",
    "❤️": " جيد ",
    "😊": " سعيد ",
    "😂": " ضحك ",
    "🔥": " ",
}

# Words whose spelling rules alone would transliterate badly.
_FRANCO_LEXICON = {
    "el": "ال",
    "al": "ال",
    "net": "نت",
    "elnet": "النت",
    "mesh": "مش",
    "msh": "مش",
    "mafeesh": "مفيش",
    "mafish": "مفيش",
    "mfesh": "مفيش",
    "wa2e3": "واقع",
    "w2e3": "واقع",
    "fatora": "فاتوره",
    "fatoora": "فاتوره",
    "fatura": "فاتوره",
    "bati2": "بطيء",
    "bate2": "بطيء",
    "ba6i2": "بطيء",
    "sor3a": "سرعه",
    "sor3t": "سرعه",
    "khat": "خط",
    "5at": "خط",
    "ta3rifa": "تعريفه",
    "tarkeeb": "تركيب",
    "tarkib": "تركيب",
    "shaghal": "شغال",
    "sha8al": "شغال",
    "sha8l": "شغال",
    "shoghl": "شغل",
    "fein": "فين",
    "feen": "فين",
    "leh": "ليه",
    "ana": "انا",
    "3ayez": "عايز",
    "3ayz": "عايز",
    "3andy": "عندي",
    "3ndy": "عندي",
    "3andi": "عندي",
    "ya": "يا",
    "w": "و",
    "wala": "ولا",
    "men": "من",
    "fe": "في",
    "fi": "في",
    "kol": "كل",
    "youm": "يوم",
    "yom": "يوم",
    "ganb": "جنب",
    "gedan": "جدا",
    "awy": "اوي",
    "awi": "اوي",
    "embare7": "امبارح",
    "emebare7": "امبارح",
    "mbare7": "امبارح",
    "sob7": "صبح",
    "shokran": "شكرا",
    "momtaza": "ممتازه",
    "gedeed": "جديد",
    "gedida": "جديده",
    "fanny": "فني",
    "fany": "فني",
    "geneh": "جنيه",
}

_FRANCO_MARKERS = frozenset(_FRANCO_LEXICON) - {"el", "al", "men", "w", "ya", "fe", "fi", "ana"}

_DIGRAPHS = (
    ("sh", "ش"),
    ("kh", "خ"),
    ("gh", "غ"),
    ("th", "ث"),
    ("dh", "ذ"),
    ("ou", "و"),
    ("ee", "ي"),
    ("oo", "و"),
)
_LETTERS = str.maketrans(
    {
        "2": "ء", "3": "ع", "5": "خ", "6": "ط", "7": "ح", "9": "ق",
        "a": "ا", "b": "ب", "c": "ك", "d": "د", "e": "ي", "f": "ف", "g": "ج", "h": "ه",
        "i": "ي", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن", "o": "و", "p": "ب",
        "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و", "v": "ف", "w": "و", "x": "كس",
        "y": "ي", "z": "ز",
    }
)  # fmt: skip

# Telecom English stays English: Arabic encoders already see these as loanwords.
_ENGLISH_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "it",
        "my",
        "your",
        "our",
        "their",
        "this",
        "that",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "by",
        "not",
        "no",
        "yes",
        "please",
        "thanks",
        "thank",
        "hi",
        "hello",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "very",
        "too",
        "so",
        "if",
        "when",
        "why",
        "how",
        "what",
        "where",
        "who",
        "internet",
        "router",
        "wifi",
        "modem",
        "bill",
        "billing",
        "invoice",
        "payment",
        "charged",
        "charge",
        "speed",
        "slow",
        "down",
        "outage",
        "service",
        "package",
        "plan",
        "install",
        "installation",
        "technician",
        "fiber",
        "fibre",
        "line",
        "dsl",
        "adsl",
        "data",
        "mobile",
        "app",
        "customer",
        "support",
        "ticket",
        "complaint",
        "refund",
        "month",
        "today",
        "yesterday",
        "days",
        "hours",
        "since",
        "still",
        "again",
        "working",
        "work",
        "works",
        "network",
        "signal",
        "connection",
        "connected",
        "disconnect",
        "disconnected",
        "problem",
        "issue",
        "help",
        "need",
        "want",
    ]
)

_NEGATION_WORDS = frozenset(
    {"مش", "مفيش", "ما", "لا", "لم", "لن", "ليس", "بدون", "مو", "not", "no", "without"}
)
_NEGATED_VERB = re.compile(r"^ما?[ء-ي]{2,}ش$")
_NON_WORD = re.compile(r"[^\w\s\[\]]", re.UNICODE)
# Franco writes the Arabic definite article as a separate word ("el net"), but Arabic attaches it.
_DETACHED_ARTICLE = re.compile(r"(?<!\S)ال (?=[ء-ي])")


def detect_script(text: str) -> Script:
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if arabic == 0 and latin == 0:
        return Script.ARABIC
    if arabic and latin / (arabic + latin) > 0.15:
        return Script.MIXED
    if arabic:
        return Script.ARABIC
    tokens = [t.strip(".,!?;:()\"'").lower() for t in text.split()]
    if any(_FRANCO_DIGIT_WORD.match(t) or t in _FRANCO_MARKERS for t in tokens):
        return Script.FRANCO
    return Script.ENGLISH


def _transliterate_word(word: str) -> str:
    if word in _FRANCO_LEXICON:
        return _FRANCO_LEXICON[word]
    prefix = ""
    if word.startswith("el") and len(word) > 4:
        prefix, word = "ال", word[2:]
        if word in _FRANCO_LEXICON:
            return prefix + _FRANCO_LEXICON[word].removeprefix("ال")
    for latin, arabic in _DIGRAPHS:
        word = word.replace(latin, arabic)
    return prefix + word.translate(_LETTERS)


def _normalize_franco_tokens(text: str, script: Script) -> str:
    output: list[str] = []
    for token in _TOKEN.findall(text):
        if token.startswith("[") and token.endswith("]"):
            output.append(token)
            continue
        lead, core, trail = _PUNCTUATION_EDGES.match(token).groups()
        should_map = bool(_FRANCO_DIGIT_WORD.match(core)) or core in _FRANCO_MARKERS
        if script is Script.FRANCO and core.isalnum() and not core.isdigit():
            should_map = should_map or core not in _ENGLISH_WORDS
        output.append(f"{lead}{_transliterate_word(core) if should_map else core}{trail}")
    return _DETACHED_ARTICLE.sub("ال", " ".join(output))


def normalize(text: str) -> str:
    """Input must already be PII-masked, so [phone]-style tokens pass through untouched."""
    normalized = to_latin_digits(text)
    normalized = _URL.sub(" [url] ", normalized)
    for emoji, replacement in _EMOJI_TOKENS.items():
        normalized = normalized.replace(emoji, replacement)
    normalized = _DIACRITICS.sub("", normalized).replace(_TATWEEL, "")
    normalized = normalized.translate(_ARABIC_UNIFY)
    normalized = _REPEATED.sub(r"\1", normalized.lower())
    script = detect_script(normalized)
    if script in (Script.FRANCO, Script.MIXED):
        normalized = _normalize_franco_tokens(normalized, script)
    return _WHITESPACE.sub(" ", normalized).strip()


def embedding_text(normalized_text: str) -> str:
    """Strip punctuation before embedding. The multilingual model is very sensitive to it in Arabic:
    adding "!!" dropped cosine similarity to about 0.4 for an otherwise identical sentence."""
    without_punctuation = _NON_WORD.sub(" ", normalized_text)
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def negation_signature(normalized_text: str) -> str:
    """Count negations. Two texts with different counts can mean opposite things, so a semantic
    cache hit must match on this value as well as on embedding similarity."""
    count = 0
    for token in normalized_text.split():
        word = token.strip(".,!?؟،؛:()\"'")
        if word in _NEGATION_WORDS or _NEGATED_VERB.match(word):
            count += 1
    return str(count)
