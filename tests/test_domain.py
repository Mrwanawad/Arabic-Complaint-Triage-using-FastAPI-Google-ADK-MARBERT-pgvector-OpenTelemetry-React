import pytest

from triage.domain.models import Category, Complaint, Prediction, Script, Sentiment, Urgency
from triage.domain.normalization import (
    detect_script,
    embedding_text,
    negation_signature,
    normalize,
)
from triage.domain.pii import mask_pii
from triage.domain.routing import OUTAGE_PRIORITY_QUEUE, route


def prediction(category=Category.BILLING, urgency=Urgency.NORMAL) -> Prediction:
    return Prediction(category, 0.9, urgency, 0.9, Sentiment.NEGATIVE, 0.9, "test")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("النت واقع من امبارح", Script.ARABIC),
        ("el net wa2e3 men embare7", Script.FRANCO),
        ("mesh shaghal", Script.FRANCO),
        ("my internet is down since yesterday", Script.ENGLISH),
        ("الراوتر router مش شغال خالص working", Script.MIXED),
    ],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected


def test_normalize_unifies_arabic_forms():
    assert normalize("أنا إيه آخر فاتورة") == "انا ايه اخر فاتوره"


def test_normalize_strips_diacritics_and_tatweel():
    assert normalize("النَّـــت") == "النت"


def test_normalize_collapses_repeated_letters():
    assert normalize("بطييييء جداااا") == "بطيء جدا"


def test_normalize_keeps_double_letters():
    assert "الله" in normalize("الله")


def test_normalize_maps_emoji_and_url():
    result = normalize("النت واقع 😡 https://example.com/x")
    assert "غاضب" in result
    assert "[url]" in result
    assert "example" not in result


def test_normalize_transliterates_franco():
    assert normalize("el net wa2e3") == "النت واقع"


def test_normalize_franco_matches_arabic_spelling():
    assert normalize("el net wa2e3 men embare7") == normalize("النت واقع من امبارح")


def test_normalize_keeps_english_in_franco():
    assert "router" in normalize("el router mesh shaghal")


def test_normalize_preserves_pii_tokens():
    assert "[phone]" in normalize(mask_pii("el net wa2e3 01012345678"))


def test_normalize_leaves_plain_english_alone():
    assert normalize("My internet is down") == "my internet is down"


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("رقمي 01012345678", "[phone]"),
        ("رقمي 010 1234 5678", "[phone]"),
        ("call +201012345678 now", "[phone]"),
        ("رقمي ٠١٠١٢٣٤٥٦٧٨", "[phone]"),
        ("الرقم القومي 29801011234567", "[national_id]"),
        ("mail me at a.b+c@example.com", "[email]"),
        ("رقم الحساب 123456789", "[number]"),
        ("التليفون الارضي 0212345678", "[phone]"),
    ],
)
def test_mask_pii(text, token):
    masked = mask_pii(text)
    assert token in masked
    assert not any(run in masked for run in ("0101234", "2980101", "123456789", "@example"))


def test_mask_pii_keeps_surrounding_spaces():
    assert mask_pii("رقمي 01012345678 شكرا") == "رقمي [phone] شكرا"
    assert mask_pii("call +20 1012345678 now") == "call [phone] now"


def test_mask_pii_customer_id():
    assert mask_pii("حسابي ABC-77 واقف", "ABC-77") == "حسابي [customer_id] واقف"


def test_mask_pii_keeps_short_numbers():
    assert mask_pii("باقة 100 جيجا") == "باقة 100 جيجا"


def test_negation_signature_separates_opposites():
    assert negation_signature(normalize("النت شغال")) != negation_signature(
        normalize("النت مش شغال")
    )


def test_negation_signature_detects_negated_verb():
    assert negation_signature(normalize("النت ماشتغلش")) == "1"


def test_embedding_text_ignores_punctuation_and_keeps_pii_tokens():
    assert embedding_text("النت واقع!!") == embedding_text("النت واقع")
    assert embedding_text("الفاتوره، غلط؟") == "الفاتوره غلط"
    assert embedding_text("رقمي [national_id] ...") == "رقمي [national_id]"


def test_ticket_id_deterministic_with_external_id():
    a = Complaint(text="x", channel="whatsapp", external_id="42")
    b = Complaint(text="y", channel="whatsapp", external_id="42")
    c = Complaint(text="x", channel="social", external_id="42")
    assert a.ticket_id == b.ticket_id
    assert a.ticket_id != c.ticket_id


def test_route_outage_high_goes_to_priority():
    assert route(prediction(Category.OUTAGE, Urgency.HIGH)) == OUTAGE_PRIORITY_QUEUE


def test_route_normal():
    assert route(prediction()) == "billing"


def test_route_high_billing_is_priority():
    assert route(prediction(urgency=Urgency.HIGH)) == "billing_priority"


def test_confidence_is_weakest_routing_head():
    p = Prediction(Category.OUTAGE, 0.95, Urgency.HIGH, 0.4, Sentiment.NEGATIVE, 0.99, "t")
    assert p.confidence == 0.4
