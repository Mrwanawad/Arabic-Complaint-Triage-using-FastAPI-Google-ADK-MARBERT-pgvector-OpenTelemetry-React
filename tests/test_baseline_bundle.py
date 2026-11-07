import pickle

import pytest

pytest.importorskip("imblearn")
pytest.importorskip("pandas")

from training.train_baseline import _model, serving_pipeline

TEXTS = ["النت واقع", "الفاتوره غلط", "النت بطيء", "الخط فاصل", "الفلوس اتخصمت", "السرعه وحشه"] * 3
LABELS = ["outage", "billing", "speed", "outage", "billing", "speed"] * 3


@pytest.mark.parametrize("strategy", ["class_weight", "oversample"])
def test_saved_model_does_not_need_imbalanced_learn(strategy):
    """The production image has no imbalanced-learn, so a pickle naming it crashes at startup."""
    fitted = _model(strategy).fit(TEXTS, LABELS)
    served = serving_pipeline(fitted)
    assert b"imblearn" not in pickle.dumps(served)
    assert list(served.classes_) == sorted(set(LABELS))
    assert served.predict_proba(["النت واقع"]).shape == (1, 3)
    assert (served.predict(TEXTS) == fitted.predict(TEXTS)).all()
