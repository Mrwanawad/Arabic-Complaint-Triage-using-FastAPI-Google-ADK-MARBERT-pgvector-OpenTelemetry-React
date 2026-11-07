"""PII masking. Runs before any external provider call or cache write.

Regex masking cannot catch personal names in free text. See Security & Privacy in architecture.md.
"""

from __future__ import annotations

import re

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NATIONAL_ID = re.compile(r"(?<!\d)[23]\d{13}(?!\d)")
_MOBILE = re.compile(r"(?<!\d)(?:(?:\+?20|0020)[\s-]?)?0?1[0125][\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_LANDLINE = re.compile(r"(?<!\d)0(?:2|3)[\s-]?\d{7,8}(?!\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{7,}(?!\d)")


def to_latin_digits(text: str) -> str:
    return text.translate(_ARABIC_DIGITS)


def mask_pii(text: str, customer_id: str | None = None) -> str:
    """Replace personal identifiers with stable tokens. Order runs from most to least specific."""
    masked = to_latin_digits(text)
    if customer_id:
        masked = re.sub(re.escape(to_latin_digits(customer_id)), "[customer_id]", masked)
    masked = _EMAIL.sub("[email]", masked)
    masked = _NATIONAL_ID.sub("[national_id]", masked)
    masked = _MOBILE.sub("[phone]", masked)
    masked = _LANDLINE.sub("[phone]", masked)
    return _LONG_NUMBER.sub("[number]", masked)
