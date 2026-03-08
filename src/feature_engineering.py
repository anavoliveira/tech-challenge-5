"""Feature creation and selection utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from src.utils import setup_logging

logger = setup_logging("feature_engineering")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a cleaned dataframe.

    Expects the dataframe to already have the cleaned/renamed columns
    produced by ``preprocessing.clean_data``.
    """
    df = df.copy()

    acad_cols = [c for c in ("inde", "iaa", "ieg", "ida") if c in df.columns]
    if acad_cols:
        df["indice_academico_medio"] = df[acad_cols].mean(axis=1)

    socio_cols = [c for c in ("ips", "ipv") if c in df.columns]
    if socio_cols:
        df["indice_socio_medio"] = df[socio_cols].mean(axis=1)

    grade_cols = [c for c in ("matem", "portug") if c in df.columns]
    if grade_cols:
        df["media_notas"] = df[grade_cols].mean(axis=1)

    if "inde" in df.columns and "fase" in df.columns:
        fase_median = df.groupby("fase")["inde"].transform("median")
        df["inde_desvio_fase"] = df["inde"] - fase_median

    if "anos_no_programa" in df.columns:
        df["ingressante_recente"] = (df["anos_no_programa"] <= 2).astype(int)

    _pedra_rank = {"Desconhecido": np.nan, "Ametista": 1, "\u00c1gata": 2, "Quartzo": 3, "Top\u00e1zio": 4}
    for col in ("pedra_22", "pedra_21"):
        rank_col = col.replace("pedra_", "pedra_rank_")
        if col in df.columns:
            df[rank_col] = df[col].map(_pedra_rank)

    if "pedra_rank_22" in df.columns and "pedra_rank_21" in df.columns:
        df["tendencia_pedra"] = df["pedra_rank_22"] - df["pedra_rank_21"]

    return df


def get_extended_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns available after create_features."""
    base_numeric = [
        "fase", "idade", "anos_no_programa",
        "inde", "iaa", "ieg", "ips", "ida", "ipv",
        "cg", "cf", "ct", "matem", "portug",
    ]
    derived = [
        "indice_academico_medio", "indice_socio_medio", "media_notas",
        "inde_desvio_fase", "ingressante_recente",
        "pedra_rank_22", "pedra_rank_21", "tendencia_pedra",
    ]
    return [c for c in base_numeric + derived if c in df.columns]


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 15,
) -> list[str]:
    """Use a shallow Random Forest to rank and select the top features.

    Categorical columns are label-encoded before fitting; only their
    encoded representation is used internally for importance ranking.
    """
    from sklearn.preprocessing import OrdinalEncoder

    X_enc = X.copy()
    cat_cols = X_enc.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_enc.select_dtypes(include="number").columns.tolist()

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_enc[cat_cols] = enc.fit_transform(X_enc[cat_cols].fillna("Desconhecido"))

    X_enc[num_cols] = X_enc[num_cols].fillna(X_enc[num_cols].median())

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_enc, y)

    importances = pd.Series(rf.feature_importances_, index=X_enc.columns)
    top = importances.nlargest(n_features).index.tolist()
    logger.info("Top %d features selected: %s", n_features, top)
    return top
