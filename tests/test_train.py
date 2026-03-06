"""Unit tests for src/train.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "RA": [f"RA-{i}" for i in range(n)],
        "Fase": rng.integers(0, 8, n),
        "Turma": ["A"] * n,
        "Nome": [f"Aluno-{i}" for i in range(n)],
        "Ano nasc": rng.integers(2000, 2012, n),
        "Idade 22": rng.integers(10, 20, n),
        "G\u00eanero": rng.choice(["Menino", "Menina"], n),
        "Ano ingresso": rng.integers(2015, 2022, n),
        "Institui\u00e7\u00e3o de ensino": ["Escola P\u00fablica"] * n,
        "Pedra 20": rng.choice(["Ametista", "\u00c1gata", None], n),
        "Pedra 21": rng.choice(["Ametista", "\u00c1gata", "Quartzo", None], n),
        "Pedra 22": rng.choice(["Ametista", "\u00c1gata", "Quartzo", "Top\u00e1zio"], n),
        "INDE 22": rng.uniform(3.0, 9.5, n),
        "Cg": rng.integers(100, 800, n),
        "Cf": rng.integers(0, 20, n),
        "Ct": rng.integers(0, 15, n),
        "N\u00ba Av": rng.integers(2, 4, n),
        "Avaliador1": ["Av-1"] * n,
        "Rec Av1": ["Promovido de Fase"] * n,
        "Avaliador2": ["Av-2"] * n,
        "Rec Av2": ["Mantido na Fase atual"] * n,
        "Avaliador3": [None] * n,
        "Rec Av3": [None] * n,
        "Avaliador4": [None] * n,
        "Rec Av4": [None] * n,
        "IAA": rng.uniform(5.0, 10.0, n),
        "IEG": rng.uniform(3.0, 10.0, n),
        "IPS": rng.uniform(4.0, 10.0, n),
        "Rec Psicologia": ["Sem limita\u00e7\u00f5es"] * n,
        "IDA": rng.uniform(3.0, 10.0, n),
        "Matem": rng.uniform(2.0, 10.0, n),
        "Portug": rng.uniform(2.0, 10.0, n),
        "Ingl\u00eas": [None] * n,
        "Indicado": rng.choice(["Sim", "N\u00e3o"], n),
        "Atingiu PV": rng.choice(["Sim", "N\u00e3o"], n),
        "IPV": rng.uniform(5.0, 10.0, n),
        "IAN": rng.uniform(2.0, 10.0, n),
        "Fase ideal": ["Fase 3 (7\u00ba e 8\u00ba ano)"] * n,
        "Defas": rng.integers(-3, 2, n),
        "Destaque IEG": ["Melhorar"] * n,
        "Destaque IDA": ["Melhorar"] * n,
        "Destaque IPV": ["Melhorar"] * n,
    })


# ---------------------------------------------------------------------------
# Tests: _build_candidates
# ---------------------------------------------------------------------------

class TestBuildCandidates:
    def test_returns_four_models(self):
        from src.train import _build_candidates
        candidates = _build_candidates()
        assert len(candidates) == 4

    def test_all_classifiers(self):
        from src.train import _build_candidates
        candidates = _build_candidates()
        for name, clf in candidates.items():
            assert isinstance(clf, ClassifierMixin), f"{name} is not a classifier"

    def test_expected_names(self):
        from src.train import _build_candidates
        candidates = _build_candidates()
        assert "Logistic Regression" in candidates
        assert "Random Forest" in candidates
        assert "Gradient Boosting" in candidates
        assert "XGBoost" in candidates

    def test_models_can_fit_predict(self):
        from src.train import _build_candidates
        from src.preprocessing import build_preprocessor, clean_data, ALL_FEATURES, TARGET
        from src.feature_engineering import create_features
        from sklearn.pipeline import Pipeline

        df = clean_data(_make_raw_df(80))
        df = create_features(df)
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feat_cols]
        y = df[TARGET]

        candidates = _build_candidates()
        # Only test Logistic Regression (fastest) for speed
        clf = candidates["Logistic Regression"]
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# Tests: train_model (fast, patched)
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_train_model_returns_pipeline_and_metadata(self, monkeypatch, tmp_path):
        """End-to-end train with synthetic data and temp model directory."""
        import src.train as train_mod
        import src.preprocessing as prep_mod
        from src.utils import get_model_path

        # Patch data loading to return synthetic data
        monkeypatch.setattr(prep_mod, "load_raw_data", lambda: _make_raw_df(120))

        # Patch model save path to temp dir
        monkeypatch.setattr("src.utils.get_model_path", lambda: tmp_path)
        monkeypatch.setattr(train_mod, "CV_FOLDS", 2)

        pipeline, meta = train_mod.train_model()

        assert pipeline is not None
        assert "model_name" in meta
        assert "feature_columns" in meta
        assert "test_metrics" in meta

    def test_train_model_saves_file(self, monkeypatch, tmp_path):
        import src.train as train_mod
        import src.preprocessing as prep_mod

        monkeypatch.setattr(prep_mod, "load_raw_data", lambda: _make_raw_df(120))
        monkeypatch.setattr("src.utils.get_model_path", lambda: tmp_path)
        monkeypatch.setattr(train_mod, "CV_FOLDS", 2)

        train_mod.train_model()

        assert (tmp_path / "model.joblib").exists()

    def test_train_model_metadata_has_metrics(self, monkeypatch, tmp_path):
        import src.train as train_mod
        import src.preprocessing as prep_mod

        monkeypatch.setattr(prep_mod, "load_raw_data", lambda: _make_raw_df(120))
        monkeypatch.setattr("src.utils.get_model_path", lambda: tmp_path)
        monkeypatch.setattr(train_mod, "CV_FOLDS", 2)

        _, meta = train_mod.train_model()

        metrics = meta["test_metrics"]
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
