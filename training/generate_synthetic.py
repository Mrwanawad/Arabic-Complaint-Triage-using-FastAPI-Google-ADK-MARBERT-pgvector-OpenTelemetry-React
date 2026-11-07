"""Generate a SYNTHETIC starter dataset from hand-written templates.

Metrics on this data only prove the pipeline runs. Template data is far easier than real complaints,
so scores here say nothing about production accuracy. Replace it with labeled real tickets.

Usage: python -m training.generate_synthetic --per-combo 40 --out data/synthetic.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# (text, category, urgency, sentiment, script)
TEMPLATES: list[tuple[str, str, str, str, str]] = [
    # outage
    ("النت واقع من {time} ومفيش خدمة خالص", "outage", "high", "negative", "arabic"),
    ("الخدمة قاطعة في {area} كلها من {time}", "outage", "high", "negative", "arabic"),
    ("مفيش نت في البيت من {time} والشغل واقف", "outage", "high", "negative", "arabic"),
    ("الخط فاصل تماما وبكلم خدمة العملاء من {time}", "outage", "high", "negative", "arabic"),
    ("هو في عطل في {area}؟ النت فصل من شوية", "outage", "normal", "neutral", "arabic"),
    ("el net wa2e3 men {time_f} w mafeesh ay service", "outage", "high", "negative", "franco"),
    ("el 5at fasel fi {area_f} kolaha men {time_f}", "outage", "high", "negative", "franco"),
    ("mafeesh net khales men {time_f} w sho8ly wa2ef", "outage", "high", "negative", "franco"),
    ("howa fe 3otl fi {area_f}? el net fasal", "outage", "normal", "neutral", "franco"),
    (
        "internet is completely down in {area_en} since {time_en}",
        "outage",
        "high",
        "negative",
        "english",
    ),
    ("no service at all since {time_en}, please fix it", "outage", "high", "negative", "english"),
    (
        "is there an outage in {area_en}? my connection just dropped",
        "outage",
        "normal",
        "neutral",
        "english",
    ),
    ("الراوتر router مفيهوش signal والنت واقع من {time}", "outage", "high", "negative", "mixed"),
    ("الـ internet قاطع في {area} من {time}", "outage", "high", "negative", "mixed"),
    # billing
    ("الفاتورة جت {amount} جنيه وأنا باقتي أقل من كده", "billing", "normal", "negative", "arabic"),
    ("اتخصم مني {amount} جنيه من غير سبب", "billing", "high", "negative", "arabic"),
    ("عايز أعرف تفاصيل فاتورة الشهر ده", "billing", "normal", "neutral", "arabic"),
    ("دفعت الفاتورة ولسه الخدمة موقوفة بسبب المديونية", "billing", "high", "negative", "arabic"),
    ("شكرا تم استرجاع الفلوس اللي اتخصمت بالغلط", "billing", "normal", "positive", "arabic"),
    ("el fatora gat {amount} geneh w ba2ty a2al", "billing", "normal", "negative", "franco"),
    ("etkhasam meny {amount} geneh men 8er sabab", "billing", "high", "negative", "franco"),
    ("3ayez a3raf tafaseel fatoret el shahr", "billing", "normal", "neutral", "franco"),
    ("shokran el felos rege3et", "billing", "normal", "positive", "franco"),
    ("i was charged {amount} EGP twice this month", "billing", "high", "negative", "english"),
    ("please explain the charges on my latest invoice", "billing", "normal", "neutral", "english"),
    ("thanks, the refund arrived quickly", "billing", "normal", "positive", "english"),
    (
        "الـ invoice بتاعة الشهر ده فيها {amount} جنيه زيادة",
        "billing",
        "normal",
        "negative",
        "mixed",
    ),
    ("عملت payment ولسه الـ service موقوفة", "billing", "high", "negative", "mixed"),
    # speed
    ("النت بطيء جدا من {time} ومش عارف أشتغل", "speed", "normal", "negative", "arabic"),
    ("السرعة أقل بكتير من الباقة اللي دافع فيها", "speed", "normal", "negative", "arabic"),
    ("النت بيقطع كل شوية والسرعة وحشة", "speed", "normal", "negative", "arabic"),
    ("عايز أزود سرعة الباقة، إيه العروض؟", "speed", "normal", "neutral", "arabic"),
    ("السرعة بقت ممتازة بعد التحديث، شكرا", "speed", "normal", "positive", "arabic"),
    ("el net bate2 awy men {time_f}", "speed", "normal", "negative", "franco"),
    ("el sor3a a2al men el ba2a elly dafe3 feha", "speed", "normal", "negative", "franco"),
    ("el net bey2ta3 kol shwaya w el sor3a we7sha", "speed", "normal", "negative", "franco"),
    ("el sor3a ba2et momtaza, shokran", "speed", "normal", "positive", "franco"),
    ("my internet speed is very slow since {time_en}", "speed", "normal", "negative", "english"),
    ("i pay for 100 mbps but only get 5", "speed", "normal", "negative", "english"),
    ("how can i upgrade my speed plan?", "speed", "normal", "neutral", "english"),
    ("الـ speed وحشة جدا والـ wifi بيفصل", "speed", "normal", "negative", "mixed"),
    ("الباقة 100 mbps والسرعة الحقيقية 5 بس", "speed", "normal", "negative", "mixed"),
    # installation
    ("الفني ما جاش في معاد التركيب", "installation", "normal", "negative", "arabic"),
    ("طلبت تركيب خط جديد من {time} ومحدش رد", "installation", "normal", "negative", "arabic"),
    ("عايز أنقل الخط لشقة جديدة في {area}", "installation", "normal", "neutral", "arabic"),
    ("الفني ركب الراوتر بسرعة وكان محترم جدا", "installation", "normal", "positive", "arabic"),
    ("عندي مناسبة ومحتاج التركيب النهارده ضروري", "installation", "high", "negative", "arabic"),
    ("el fanny magash fi me3ad el tarkeeb", "installation", "normal", "negative", "franco"),
    (
        "talabt tarkeeb 5at gedeed men {time_f} w ma7adesh rad",
        "installation",
        "normal",
        "negative",
        "franco",
    ),
    ("3ayez an2el el 5at le sha2a fi {area_f}", "installation", "normal", "neutral", "franco"),
    (
        "the technician never showed up for my installation",
        "installation",
        "normal",
        "negative",
        "english",
    ),
    (
        "i want to move my line to a new apartment in {area_en}",
        "installation",
        "normal",
        "neutral",
        "english",
    ),
    (
        "installation was quick and the technician was great",
        "installation",
        "normal",
        "positive",
        "english",
    ),
    (
        "الـ technician ماجاش ولسه مستني الـ installation",
        "installation",
        "normal",
        "negative",
        "mixed",
    ),
    ("عايز fiber جديد في {area} إيه الإجراءات؟", "installation", "normal", "neutral", "mixed"),
    # other
    ("عايز أعرف مواعيد عمل الفرع في {area}", "other", "normal", "neutral", "arabic"),
    ("الموظف كلمني بطريقة وحشة جدا", "other", "normal", "negative", "arabic"),
    ("خدمة العملاء ممتازة، شكرا ليكم", "other", "normal", "positive", "arabic"),
    ("حد سرق بياناتي وبيستخدم خطي من غير علمي", "other", "high", "negative", "arabic"),
    ("3ayez a3raf mawa3eed el fara3 fi {area_f}", "other", "normal", "neutral", "franco"),
    ("el mowazaf kalemny we7esh awy", "other", "normal", "negative", "franco"),
    ("khedmet el 3omala momtaza, shokran", "other", "normal", "positive", "franco"),
    ("what are the branch opening hours in {area_en}?", "other", "normal", "neutral", "english"),
    ("your support agent was rude to me", "other", "normal", "negative", "english"),
    ("someone is using my line without permission", "other", "high", "negative", "english"),
    ("great customer support, thank you", "other", "normal", "positive", "english"),
    ("الـ agent كان محترم والـ support ممتاز", "other", "normal", "positive", "mixed"),
    ("فين أقرب branch في {area}؟", "other", "normal", "neutral", "mixed"),
]

SLOTS = {
    "time": ["امبارح", "الصبح", "يومين", "أسبوع", "ساعتين", "إمبارح بالليل"],
    "time_f": ["embare7", "el sob7", "yomeen", "osboo3", "sa3teen"],
    "time_en": ["yesterday", "this morning", "two days", "last week", "two hours"],
    "area": ["المعادي", "مدينة نصر", "الإسكندرية", "المهندسين", "طنطا", "أسيوط"],
    "area_f": ["el ma3adi", "madinet nasr", "eskendereya", "el mohandeseen", "tanta"],
    "area_en": ["Maadi", "Nasr City", "Alexandria", "Mohandessin", "Tanta"],
    "amount": ["300", "450", "700", "1200", "85"],
}

NOISE = ["", "", "", " 😡", "!!", " بسرعة لو سمحت", " 🙏", "..."]


def _fill(template: str, rng: random.Random) -> str:
    text = template
    for slot, values in SLOTS.items():
        token = "{" + slot + "}"
        while token in text:
            text = text.replace(token, rng.choice(values), 1)
    return text + rng.choice(NOISE)


def generate(per_combo: int, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for group, (template, category, urgency, sentiment, script) in enumerate(TEMPLATES):
        for _ in range(per_combo):
            text = _fill(template, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append(
                {
                    "text": text,
                    "category": category,
                    "urgency": urgency,
                    "sentiment": sentiment,
                    "script": script,
                    # Splits must group by template, or near-duplicates leak into the test set.
                    "template_id": str(group),
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-combo", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    args = parser.parse_args()

    rows = generate(args.per_combo, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
