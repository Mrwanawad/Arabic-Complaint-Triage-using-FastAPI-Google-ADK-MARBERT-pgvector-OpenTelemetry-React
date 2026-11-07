import asyncio

import pytest

from tests.fakes import (
    FakeCache,
    FakeClassifier,
    FakeEmbedder,
    FakeEscalator,
    FakeExamples,
    FakeTickets,
    FakeTranscriber,
    FakeUrgency,
    make_prediction,
)
from triage.application.use_cases import (
    ClassifyComplaint,
    IngestVoiceNote,
    ReviewTicket,
    UnknownModel,
)
from triage.domain.models import (
    Category,
    Complaint,
    DecidedBy,
    LabeledExample,
    Sentiment,
    TicketStatus,
    Urgency,
)
from triage.domain.routing import HUMAN_REVIEW_QUEUE, OUTAGE_PRIORITY_QUEUE

LOW = make_prediction(0.4)


def build(classifier=None, **kwargs):
    kwargs.setdefault("tickets", FakeTickets())
    return ClassifyComplaint(
        classifiers={"encoder": classifier or FakeClassifier(), "baseline": FakeClassifier()},
        default_model="encoder",
        **kwargs,
    )


def run(use_case, complaint=None, **kwargs):
    return asyncio.run(use_case.execute(complaint or Complaint(text="النت واقع"), **kwargs))


def test_confident_encoder_skips_llm():
    escalator = FakeEscalator()
    ticket = run(build(escalator=escalator))
    assert ticket.decided_by is DecidedBy.ENCODER
    assert ticket.target_queue == OUTAGE_PRIORITY_QUEUE
    assert ticket.status is TicketStatus.ROUTED
    assert escalator.calls == []


def test_low_confidence_escalates_to_llm_and_caches():
    escalator, cache = FakeEscalator(), FakeCache()
    ticket = run(
        build(
            FakeClassifier(LOW),
            escalator=escalator,
            embedder=FakeEmbedder(),
            cache=cache,
            examples=FakeExamples(
                [LabeledExample("x", Category.OUTAGE, Urgency.HIGH, Sentiment.NEGATIVE)]
            ),
        )
    )
    assert ticket.decided_by is DecidedBy.LLM
    assert ticket.prediction.category is Category.BILLING
    assert len(escalator.calls[0][1]) == 1
    assert len(cache.stored) == 1


def test_cache_hit_skips_llm():
    escalator = FakeEscalator()
    ticket = run(
        build(
            FakeClassifier(LOW),
            escalator=escalator,
            embedder=FakeEmbedder(),
            cache=FakeCache(hit=make_prediction(0.9, Category.SPEED)),
        )
    )
    assert ticket.decided_by is DecidedBy.CACHE
    assert ticket.prediction.category is Category.SPEED
    assert escalator.calls == []


def test_llm_receives_masked_text_only():
    escalator = FakeEscalator()
    run(
        build(FakeClassifier(LOW), escalator=escalator),
        Complaint(text="النت واقع رقمي 01012345678"),
    )
    sent = escalator.calls[0][0]
    assert "01012345678" not in sent
    assert "[phone]" in sent


def test_all_providers_down_goes_to_human_with_encoder_result():
    ticket = run(build(FakeClassifier(LOW), escalator=FakeEscalator(fail=True)))
    assert ticket.status is TicketStatus.PENDING_REVIEW
    assert ticket.target_queue == HUMAN_REVIEW_QUEUE
    assert ticket.review_reason == "llm_unavailable"
    assert ticket.decided_by is DecidedBy.ENCODER


def test_unsure_llm_goes_to_human_and_is_not_cached():
    cache = FakeCache()
    ticket = run(
        build(
            FakeClassifier(LOW),
            escalator=FakeEscalator(make_prediction(0.5)),
            embedder=FakeEmbedder(),
            cache=cache,
        )
    )
    assert ticket.status is TicketStatus.PENDING_REVIEW
    assert ticket.review_reason == "low_confidence_llm"
    assert cache.stored == []


def test_no_llm_configured_goes_to_human():
    ticket = run(build(FakeClassifier(LOW)))
    assert ticket.review_reason == "low_confidence_no_llm"


def test_cache_outage_does_not_lose_complaint():
    cache = FakeCache()
    cache.fail = True
    ticket = run(
        build(FakeClassifier(LOW), escalator=FakeEscalator(), embedder=FakeEmbedder(), cache=cache)
    )
    assert ticket.decided_by is DecidedBy.LLM


def test_replay_returns_stored_ticket_without_inference():
    tickets = FakeTickets()
    classifier = FakeClassifier()
    use_case = build(classifier, tickets=tickets)
    complaint = Complaint(text="النت واقع", channel="whatsapp", external_id="m1")
    first = run(use_case, complaint)
    second = run(use_case, complaint)
    assert first.ticket_id == second.ticket_id
    assert len(classifier.calls) == 1


def test_baseline_selection_reports_baseline():
    assert run(build(), model="baseline").decided_by is DecidedBy.BASELINE


def test_unknown_model_rejected():
    with pytest.raises(UnknownModel):
        run(build(), model="nope")


def test_urgency_model_overrides_and_gets_repeat_count():
    urgency = FakeUrgency()
    ticket = run(
        build(tickets=FakeTickets(repeats=3), urgency_model=urgency),
        Complaint(text="النت واقع", customer_id="c1"),
    )
    assert ticket.prediction.urgency is Urgency.NORMAL
    assert ticket.prediction.urgency_confidence == 0.77
    assert urgency.features.repeat_complaints_24h == 3


def test_voice_note_flows_through_classification():
    use_case = IngestVoiceNote(FakeTranscriber(), build())
    ticket = asyncio.run(use_case.execute("النت واقع".encode()))
    assert ticket.channel == "voice"
    assert ticket.raw_text == "النت واقع"


def test_review_applies_decision_and_feeds_examples():
    tickets, examples = FakeTickets(), FakeExamples()
    pending = run(build(FakeClassifier(LOW), tickets=tickets))
    review = ReviewTicket(tickets, FakeEmbedder(), examples)
    updated = asyncio.run(
        review.execute(
            pending.ticket_id, Category.SPEED, Urgency.NORMAL, Sentiment.NEUTRAL, "agent-1"
        )
    )
    assert updated.status is TicketStatus.REVIEWED
    assert updated.decided_by is DecidedBy.HUMAN
    assert updated.target_queue == "technical_support"
    assert examples.items[0].category is Category.SPEED


def test_review_unknown_ticket_returns_none():
    import uuid

    review = ReviewTicket(FakeTickets())
    assert (
        asyncio.run(
            review.execute(uuid.uuid4(), Category.OTHER, Urgency.NORMAL, Sentiment.NEUTRAL, "a")
        )
        is None
    )
