"""p_choice from a trained models/model.json, for live / collect / offline (`--model`)."""
from __future__ import annotations

from gazedemo.core.choice_model import ConditionalLogit
from gazedemo.core.features import feature_matrix, plate_rows
from gazedemo.io.logs import read_json
from gazedemo.types import TrialResult


class ChoiceScorer:
    def __init__(self, path, plates):
        data = read_json(path)
        self.model = ConditionalLogit.from_json(data)
        self.plates = list(plates)
        self.info = {k: data.get(k) for k in ("feature_set", "n_trial", "trained_at")}
        model_plates = data.get("plates")
        if model_plates and list(model_plates) != self.plates:
            print(f"[model] WARNING: model trained with plates {model_plates}, running with {self.plates}")

    def predict(self, res: TrialResult) -> dict[str, float]:
        liking = {p: res.meta.get(f"liking_{p}") for p in self.plates}
        X = feature_matrix(plate_rows(res, self.plates, liking), self.model.feature_names)
        return dict(zip(self.plates, (float(p) for p in self.model.predict_proba(X))))
