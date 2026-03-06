"""Unit tests for src/preprocessing.py (>=80% coverage target)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PEDRA_ORDER,
    TARGET,
    _resolve_genero_column,
    build_preprocessor,
    clean_data,
    prepare_dataset,
)
from src.utils import get_database_path, load_model, save_model, setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(n: int = 10) -> pd.DataFrame:
    """Minimal synthetic dataframe that mirrors the raw Excel structure."""
    rng = np.random.default_rng(42)
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
# Tests: clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_returns_dataframe(self):
        df = clean_data(_make_raw_df())
        assert isinstance(df, pd.DataFrame)

    def test_target_column_created(self):
        df = clean_data(_make_raw_df())
        assert TARGET in df.columns

    def test_target_is_binary(self):
        df = clean_data(_make_raw_df(50))
        assert set(df[TARGET].unique()).issubset({0, 1})

    def test_anos_no_programa_positive(self):
        df = clean_data(_make_raw_df())
        assert (df["anos_no_programa"] > 0).all()

    def test_pedra_nulls_filled(self):
        df = clean_data(_make_raw_df(20))
        assert df["pedra_22"].isna().sum() == 0
        assert df["pedra_21"].isna().sum() == 0

    def test_grade_nulls_filled(self):
        raw = _make_raw_df(20)
        raw.loc[0, "Matem"] = None
        raw.loc[1, "Portug"] = None
        df = clean_data(raw)
        assert df["matem"].isna().sum() == 0
        assert df["portug"].isna().sum() == 0

    def test_does_not_modify_original(self):
        raw = _make_raw_df()
        original_cols = list(raw.columns)
        clean_data(raw)
        assert list(raw.columns) == original_cols

    def test_genero_column_present(self):
        df = clean_data(_make_raw_df())
        assert "genero" in df.columns

    def test_handles_missing_pedra_21_column(self):
        """When pedra_21 column is missing, it should be filled with Desconhecido."""
        raw = _make_raw_df(10)
        raw = raw.drop(columns=["Pedra 21"], errors="ignore")
        df = clean_data(raw)
        assert "pedra_21" in df.columns
        assert (df["pedra_21"] == "Desconhecido").all()

    def test_handles_missing_ano_ingresso(self):
        """When Ano ingresso column is missing, anos_no_programa should be 0."""
        raw = _make_raw_df(5)
        raw = raw.drop(columns=["Ano ingresso"], errors="ignore")
        df = clean_data(raw)
        assert (df["anos_no_programa"] == 0).all()


# ---------------------------------------------------------------------------
# Tests: _resolve_genero_column
# ---------------------------------------------------------------------------

class TestResolveGeneroColumn:
    def test_resolves_unicode_column(self):
        df = pd.DataFrame({"G\u00eanero": ["Menino", "Menina"]})
        result = _resolve_genero_column(df)
        assert "genero" in result.columns

    def test_resolves_plain_column(self):
        df = pd.DataFrame({"Genero": ["Menino", "Menina"]})
        result = _resolve_genero_column(df)
        assert "genero" in result.columns

    def test_creates_dummy_when_missing(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = _resolve_genero_column(df)
        assert "genero" in result.columns
        assert (result["genero"] == "Desconhecido").all()


# ---------------------------------------------------------------------------
# Tests: create_target (via clean_data)
# ---------------------------------------------------------------------------

class TestCreateTarget:
    def test_defas_zero_is_risk(self):
        raw = _make_raw_df(5)
        raw["Defas"] = [0, -1, 1, -2, 2]
        df = clean_data(raw)
        expected = [1, 0, 1, 0, 1]
        assert list(df[TARGET]) == expected

    def test_negative_defas_is_no_risk(self):
        raw = _make_raw_df(3)
        raw["Defas"] = [-1, -2, -3]
        df = clean_data(raw)
        assert df[TARGET].sum() == 0

    def test_positive_defas_is_risk(self):
        raw = _make_raw_df(3)
        raw["Defas"] = [1, 2, 3]
        df = clean_data(raw)
        assert df[TARGET].sum() == 3


# ---------------------------------------------------------------------------
# Tests: build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        prep = build_preprocessor()
        assert isinstance(prep, ColumnTransformer)

    def test_fit_transform_shape(self):
        df = clean_data(_make_raw_df(30))
        available = [c for c in ALL_FEATURES if c in df.columns]
        X = df[available]
        prep = build_preprocessor()
        Xt = prep.fit_transform(X)
        assert Xt.shape[0] == 30

    def test_no_nan_after_transform(self):
        df = clean_data(_make_raw_df(30))
        available = [c for c in ALL_FEATURES if c in df.columns]
        X = df[available]
        prep = build_preprocessor()
        Xt = prep.fit_transform(X)
        assert not np.isnan(Xt).any()

    def test_transform_on_new_data(self):
        """Fitted preprocessor should transform unseen data without errors."""
        df_train = clean_data(_make_raw_df(30))
        df_test = clean_data(_make_raw_df(10))
        available = [c for c in ALL_FEATURES if c in df_train.columns]
        prep = build_preprocessor()
        prep.fit(df_train[available])
        Xt = prep.transform(df_test[available])
        assert Xt.shape[0] == 10


# ---------------------------------------------------------------------------
# Tests: prepare_dataset (integration, reads real file)
# ---------------------------------------------------------------------------

class TestPrepareDataset:
    def test_prepare_dataset_returns_x_y(self):
        """Integration test — requires database/base_2024.xlsx to exist."""
        db_path = get_database_path() / "base_2024.xlsx"
        if not db_path.exists():
            pytest.skip("Database file not available")
        X, y = prepare_dataset()
        assert len(X) > 0
        assert len(y) == len(X)

    def test_prepare_dataset_binary_target(self):
        db_path = get_database_path() / "base_2024.xlsx"
        if not db_path.exists():
            pytest.skip("Database file not available")
        X, y = prepare_dataset()
        assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_pedra_order_length(self):
        assert len(PEDRA_ORDER) == 5

    def test_all_features_non_empty(self):
        assert len(ALL_FEATURES) > 0

    def test_feature_lists_disjoint(self):
        assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)

    def test_target_string(self):
        assert TARGET == "risco"


# ---------------------------------------------------------------------------
# Tests: utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_setup_logging_returns_logger(self):
        import logging
        logger = setup_logging("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice should not duplicate handlers."""
        logger1 = setup_logging("same_name")
        handler_count = len(logger1.handlers)
        logger2 = setup_logging("same_name")
        assert len(logger2.handlers) == handler_count

    def test_get_database_path_exists(self):
        assert get_database_path().exists()

    def test_save_and_load_model(self, tmp_path, monkeypatch):
        """save_model + load_model round-trip."""
        from sklearn.linear_model import LogisticRegression
        from src.utils import get_model_path

        # Redirect model path to temp directory
        monkeypatch.setattr("src.utils.get_model_path", lambda: tmp_path)

        clf = LogisticRegression()
        meta = {"model_name": "test", "f1": 0.9}
        save_model(clf, meta)

        loaded_pipe, loaded_meta = load_model()
        assert loaded_meta["model_name"] == "test"
        assert loaded_meta["f1"] == 0.9

    def test_load_model_missing_raises(self, tmp_path, monkeypatch):
        """load_model should raise FileNotFoundError when no file exists."""
        monkeypatch.setattr("src.utils.get_model_path", lambda: tmp_path)
        with pytest.raises(FileNotFoundError):
            load_model()

    def test_get_model_path_returns_path(self):
        from src.utils import get_model_path
        p = get_model_path()
        assert isinstance(p, Path)
        assert p.name == "model"


# ---------------------------------------------------------------------------
# Tests: edge cases for branch coverage
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_resolve_genero_partial_match(self):
        """Cover the partial-match branch (col starts with g, ends with nero)."""
        df = pd.DataFrame({"genero_col": ["Menino"]})
        # rename to something that matches partial but not direct list
        df2 = df.rename(columns={"genero_col": "g_nero_x"})
        # Should fall through to dummy branch; won't match partial because
        # the partial match requires "nero" in col.lower()
        # This actually won't trigger line 94, so we test real partial match:
        df3 = pd.DataFrame({"g_Nero": ["Menino"]})  # starts with g, contains nero
        result = _resolve_genero_column(df3)
        assert "genero" in result.columns

    def test_prepare_dataset_with_missing_feature_triggers_warning(self, monkeypatch):
        """Cover the missing-column warning branch in prepare_dataset."""
        import src.preprocessing as pp
        original_all = pp.ALL_FEATURES[:]

        # Inject a fake feature that doesn't exist in the data
        monkeypatch.setattr("src.preprocessing.ALL_FEATURES", original_all + ["__fake__"])

        db_path = get_database_path() / "base_2024.xlsx"
        if not db_path.exists():
            pytest.skip("Database file not available")

        X, y = prepare_dataset()
        assert "__fake__" not in X.columns
