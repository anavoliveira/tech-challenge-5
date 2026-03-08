import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import setup_logging

logger = setup_logging("feature_engineering")

# ranking numérico das pedras pra calcular tendência
_pedra_rank = {
    "Desconhecido": np.nan,
    "Ametista": 1,
    "Ágata": 2,
    "Quartzo": 3,
    "Topázio": 4,
}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas a partir do dataframe limpo."""
    df = df.copy()

    # médias dos índices acadêmicos
    acad_cols = [c for c in ("inde", "iaa", "ieg", "ida") if c in df.columns]
    if acad_cols:
        df["indice_academico_medio"] = df[acad_cols].mean(axis=1)

    socio_cols = [c for c in ("ips", "ipv") if c in df.columns]
    if socio_cols:
        df["indice_socio_medio"] = df[socio_cols].mean(axis=1)

    grade_cols = [c for c in ("matem", "portug") if c in df.columns]
    if grade_cols:
        df["media_notas"] = df[grade_cols].mean(axis=1)

    # desvio do inde em relação à mediana da fase
    if "inde" in df.columns and "fase" in df.columns:
        fase_median = df.groupby("fase")["inde"].transform("median")
        df["inde_desvio_fase"] = df["inde"] - fase_median

    if "anos_no_programa" in df.columns:
        df["ingressante_recente"] = (df["anos_no_programa"] <= 2).astype(int)

    for col in ("pedra_22", "pedra_21"):
        rank_col = col.replace("pedra_", "pedra_rank_")
        if col in df.columns:
            df[rank_col] = df[col].map(_pedra_rank)

    # tendência = se o aluno subiu ou desceu de pedra no ano
    if "pedra_rank_22" in df.columns and "pedra_rank_21" in df.columns:
        df["tendencia_pedra"] = df["pedra_rank_22"] - df["pedra_rank_21"]

    return df


def get_extended_feature_columns(df: pd.DataFrame) -> list[str]:
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


def select_top_features(X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> list[str]:
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
    logger.info("top %d features: %s", n_features, top)
    return top
