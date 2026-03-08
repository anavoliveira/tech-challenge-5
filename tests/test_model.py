"""Unit tests for model training, evaluation, and API prediction logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import compare_models, evaluate_model, print_summary
from src.feature_engineering import create_features, get_extended_feature_columns, select_top_features
from src.preprocessing import (
    ALL_FEATURES,
    TARGET,
    build_preprocessor,
    clean_data,
)


 
# Helpers
def _make_raw_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


def _build_fitted_pipeline(df_raw: pd.DataFrame):
    df = clean_data(df_raw)
    df = create_features(df)
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feat_cols]
    y = df[TARGET]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


 
# Tests: feature_engineering
class TestFeatureEngineering:
    def test_create_features_adds_columns(self):
        df = clean_data(_make_raw_df(20))
        df_fe = create_features(df)
        assert "indice_academico_medio" in df_fe.columns
        assert "indice_socio_medio" in df_fe.columns
        assert "media_notas" in df_fe.columns

    def test_indice_academico_medio_range(self):
        df = clean_data(_make_raw_df(20))
        df_fe = create_features(df)
        assert df_fe["indice_academico_medio"].between(0, 10).all()

    def test_anos_no_programa_positive(self):
        df = clean_data(_make_raw_df(20))
        df_fe = create_features(df)
        assert (df_fe["anos_no_programa"] > 0).all()

    def test_inde_desvio_fase_created(self):
        df = clean_data(_make_raw_df(40))
        df_fe = create_features(df)
        assert "inde_desvio_fase" in df_fe.columns

    def test_get_extended_feature_columns(self):
        df = clean_data(_make_raw_df(20))
        df_fe = create_features(df)
        cols = get_extended_feature_columns(df_fe)
        assert len(cols) > 0
        assert all(c in df_fe.columns for c in cols)

    def test_does_not_modify_original(self):
        df = clean_data(_make_raw_df(10))
        cols_before = list(df.columns)
        create_features(df)
        assert list(df.columns) == cols_before


 
# Tests: evaluate_model
class TestEvaluateModel:
    def setup_method(self):
        raw = _make_raw_df(100)
        self.pipe, self.X, self.y = _build_fitted_pipeline(raw)

    def test_returns_dict_with_required_keys(self):
        result = evaluate_model(self.pipe, self.X, self.y)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in result

    def test_accuracy_between_0_and_1(self):
        result = evaluate_model(self.pipe, self.X, self.y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_f1_between_0_and_1(self):
        result = evaluate_model(self.pipe, self.X, self.y)
        assert 0.0 <= result["f1"] <= 1.0

    def test_roc_auc_between_0_and_1(self):
        result = evaluate_model(self.pipe, self.X, self.y)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_confusion_matrix_shape(self):
        result = evaluate_model(self.pipe, self.X, self.y)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_classification_report_is_string(self):
        result = evaluate_model(self.pipe, self.X, self.y)
        assert isinstance(result["classification_report"], str)


 
# Tests: compare_models
class TestCompareModels:
    def test_returns_best_by_f1(self):
        results = {
            "ModelA": {"f1": 0.70, "accuracy": 0.80},
            "ModelB": {"f1": 0.85, "accuracy": 0.75},
            "ModelC": {"f1": 0.60, "accuracy": 0.90},
        }
        assert compare_models(results) == "ModelB"

    def test_single_model(self):
        results = {"OnlyModel": {"f1": 0.75}}
        assert compare_models(results) == "OnlyModel"


 
# Tests: pipeline predict
 

class TestPipelinePredict:
    def setup_method(self):
        raw = _make_raw_df(100)
        self.pipe, self.X, self.y = _build_fitted_pipeline(raw)

    def test_predict_returns_array(self):
        preds = self.pipe.predict(self.X)
        assert len(preds) == len(self.X)

    def test_predict_binary_output(self):
        preds = self.pipe.predict(self.X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self):
        probas = self.pipe.predict_proba(self.X)
        assert probas.shape == (len(self.X), 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_single_sample_prediction(self):
        single = self.X.iloc[[0]]
        pred = self.pipe.predict(single)
        assert pred[0] in (0, 1)

    def test_probability_between_0_and_1(self):
        probas = self.pipe.predict_proba(self.X)
        assert (probas >= 0).all() and (probas <= 1).all()
 
# Tests: print_summary
class TestPrintSummary:
    def test_runs_without_error(self, capsys):
        results = {
            "ModelA": {"accuracy": 0.90, "precision": 0.88, "recall": 0.85, "f1": 0.86, "roc_auc": 0.93},
            "ModelB": {"accuracy": 0.85, "precision": 0.82, "recall": 0.80, "f1": 0.81, "roc_auc": None},
        }
        print_summary(results)
        captured = capsys.readouterr()
        assert "ModelA" in captured.out
        assert "ModelB" in captured.out

    def test_shows_model_names(self, capsys):
        results = {"RandomForest": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9, "roc_auc": 0.95}}
        print_summary(results)
        assert "RandomForest" in capsys.readouterr().out


 
# Tests: select_top_features
class TestSelectTopFeatures:
    def test_returns_list_of_strings(self):
        df = clean_data(_make_raw_df(60))
        df = create_features(df)
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feat_cols]
        y = df[TARGET]
        top = select_top_features(X, y, n_features=5)
        assert isinstance(top, list)
        assert all(isinstance(f, str) for f in top)

    def test_returns_correct_count(self):
        df = clean_data(_make_raw_df(60))
        df = create_features(df)
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feat_cols]
        y = df[TARGET]
        top = select_top_features(X, y, n_features=5)
        assert len(top) == 5

    def test_features_are_subset_of_input(self):
        df = clean_data(_make_raw_df(60))
        df = create_features(df)
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feat_cols]
        y = df[TARGET]
        top = select_top_features(X, y, n_features=5)
        assert all(f in X.columns for f in top)
