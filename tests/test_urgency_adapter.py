import lightgbm as lgb
import numpy as np

from tests.fakes import make_prediction
from triage.adapters.outbound.classifiers import LightGbmUrgency, urgency_feature_vector
from triage.application.ports import UrgencyFeatures
from triage.domain.models import Urgency


def trained_model(path):
    """Trains on repeat count alone, which is enough to show the feature reaches the booster."""
    rows, labels = [], []
    for repeats in range(20):
        features = UrgencyFeatures("whatsapp", 10, repeats)
        rows.append(urgency_feature_vector(make_prediction(0.9), features))
        labels.append(int(repeats >= 10))
    booster = lgb.train(
        {"objective": "binary", "min_data_in_leaf": 1, "verbosity": -1},
        lgb.Dataset(np.array(rows, dtype=np.float64), labels),
        num_boost_round=30,
    )
    booster.save_model(str(path))
    return LightGbmUrgency(path)


def test_prediction_uses_metadata_features(tmp_path):
    model = trained_model(tmp_path / "urgency.txt")
    high, high_confidence = model.predict(make_prediction(0.9), UrgencyFeatures("whatsapp", 10, 18))
    low, low_confidence = model.predict(make_prediction(0.9), UrgencyFeatures("whatsapp", 10, 1))
    assert high is Urgency.HIGH
    assert low is Urgency.NORMAL
    assert high_confidence > 0.5
    assert low_confidence > 0.5


def test_feature_vector_has_stable_order_and_encodes_channel():
    vector = urgency_feature_vector(make_prediction(0.9), UrgencyFeatures("whatsapp", 7, 2))
    assert vector[3:5] == [7.0, 2.0]
    assert len(vector) == 5 + 5 + 5
    assert sum(vector[-5:]) == 1.0
