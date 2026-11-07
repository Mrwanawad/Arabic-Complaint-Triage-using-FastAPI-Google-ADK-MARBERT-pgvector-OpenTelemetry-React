"""Measure semantic cache safety on hand-written pairs, using the real embedder.

A wrong hit reuses a decision for a message that means something else, which can misroute an outage.
Reports hit and wrong-hit rates per threshold, with and without the negation guard.

Usage: python -m evals.eval_semantic_cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triage.bootstrap import build_embedder
from triage.domain.normalization import negation_signature, normalize
from triage.domain.pii import mask_pii
from triage.settings import Settings

# A hit is correct for these: same complaint, same routing.
SAME_MEANING = [
    ("النت واقع من امبارح", "النت واقع من إمبارح!!"),
    ("النت واقع من امبارح", "النت واقع من امبارح 😡"),
    ("النت واقع من امبارح", "النت فاصل من امبارح"),
    ("النت بطيء جدا", "النت بطيء جداااا"),
    ("النت بطيء جدا", "النت بطيء اوي"),
    ("الفاتورة غلط الشهر ده", "فاتورة الشهر ده غلط"),
    ("الفني ما جاش في المعاد", "الفني ماجاش في معاده"),
    ("el net wa2e3 men embare7", "النت واقع من امبارح"),
    ("my internet is down", "my internet is down!!"),
    ("my internet is very slow", "my internet is so slow"),
    ("عايز اعرف تفاصيل الفاتورة", "عايز تفاصيل الفاتورة"),
    ("رقمي 01012345678 والنت واقع", "رقمي 01198765432 والنت واقع"),
]

# A hit is wrong for these: similar words, different meaning or routing.
DIFFERENT_MEANING = [
    ("النت شغال", "النت مش شغال"),
    ("النت واقع", "النت مش واقع"),
    ("الخدمة رجعت", "الخدمة مرجعتش"),
    ("الفني جه", "الفني ماجاش"),
    ("الفلوس اتخصمت", "الفلوس ماتخصمتش"),
    ("الفاتورة وصلت", "الفاتورة موصلتش"),
    ("internet is working", "internet is not working"),
    ("the technician came", "the technician did not come"),
    ("النت واقع", "النت بطيء"),
    ("الفاتورة غلط", "الخط غلط"),
    ("عايز اركب خط جديد", "عايز الغي الخط"),
    ("النت بطيء في المعادي", "النت واقع في المعادي"),
    ("my internet is slow", "my internet is down"),
    ("refund my payment", "confirm my payment"),
]

THRESHOLDS = (0.90, 0.93, 0.95, 0.97, 0.98, 0.99)


def prepare(text: str) -> str:
    return normalize(mask_pii(text))


def evaluate(settings: Settings) -> dict:
    embedder = build_embedder(settings)

    def score(pairs: list[tuple[str, str]]) -> list[dict]:
        rows = []
        for left, right in pairs:
            a, b = prepare(left), prepare(right)
            similarity = float(np.dot(embedder.embed(a), embedder.embed(b)))
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "similarity": round(similarity, 4),
                    "guard_passes": negation_signature(a) == negation_signature(b),
                }
            )
        return rows

    same, different = score(SAME_MEANING), score(DIFFERENT_MEANING)
    table = []
    for threshold in THRESHOLDS:
        table.append(
            {
                "threshold": threshold,
                "useful_hit_rate": round(
                    sum(r["similarity"] >= threshold and r["guard_passes"] for r in same)
                    / len(same),
                    3,
                ),
                "wrong_hits_no_guard": sum(r["similarity"] >= threshold for r in different),
                "wrong_hits_with_guard": sum(
                    r["similarity"] >= threshold and r["guard_passes"] for r in different
                ),
            }
        )
    return {
        "embedder": type(embedder).__name__,
        "same_pairs": len(same),
        "different_pairs": len(different),
        "table": table,
        "closest_wrong_pairs": sorted(different, key=lambda r: -r["similarity"])[:6],
        "weakest_same_pairs": sorted(same, key=lambda r: r["similarity"])[:4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/semantic_cache_eval.json"))
    args = parser.parse_args()
    results = evaluate(Settings())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
