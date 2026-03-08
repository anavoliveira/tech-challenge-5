"""Data loading, cleaning, and preprocessing pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils import get_database_path, setup_logging

logger = setup_logging("preprocessing")

 
# Column constants
 

# Pedra ranking from lowest to highest performance
PEDRA_ORDER = ["Desconhecido", "Ametista", "\u00c1gata", "Quartzo", "Top\u00e1zio"]

NUMERIC_FEATURES = [
    "fase",
    "idade",
    "anos_no_programa",
    "inde",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "cg",
    "cf",
    "ct",
    "matem",
    "portug",
]

CATEGORICAL_FEATURES = [
    "pedra_22",
    "pedra_21",
    "genero",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "risco"

# Mapping from raw Excel columns to internal names
_COLUMN_MAP = {
    "Fase": "fase",
    "Idade 22": "idade",
    "Ano ingresso": "ano_ingresso",
    "INDE 22": "inde",
    "IAA": "iaa",
    "IEG": "ieg",
    "IPS": "ips",
    "IDA": "ida",
    "IPV": "ipv",
    "IAN": "ian",
    "Cg": "cg",
    "Cf": "cf",
    "Ct": "ct",
    "Matem": "matem",
    "Portug": "portug",
    "Pedra 22": "pedra_22",
    "Pedra 21": "pedra_21",
    "Defas": "defas",
}

# Gênero column has special encoding; we resolve it by prefix match
_GENERO_RAW_OPTIONS = ["G\u00eanero", "Genero", "g\u00eanero", "genero"]


def load_raw_data() -> pd.DataFrame:
    """Load 2024 base Excel file."""
    path = get_database_path() / "base_2024.xlsx"
    df = pd.read_excel(path)
    logger.info("Loaded %s: %s", path.name, df.shape)
    return df


def _resolve_genero_column(df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
    """Rename the gender column regardless of encoding."""
    for candidate in _GENERO_RAW_OPTIONS:
        if candidate in df.columns:
            return df.rename(columns={candidate: "genero"})
    # Try partial match
    for col in df.columns:
        if col.lower().startswith("g") and "nero" in col.lower():
            return df.rename(columns={col: "genero"})
    logger.warning("Gender column not found; creating dummy.")
    df["genero"] = "Desconhecido"
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, engineer base features, create binary target."""
    df = df.copy()

    # Rename known columns
    df = df.rename(columns=_COLUMN_MAP)

    # Resolve gender column
    df = _resolve_genero_column(df)

    # Derived feature: years in programme
    if "ano_ingresso" in df.columns:
        df["anos_no_programa"] = 2022 - df["ano_ingresso"] + 1
    else:
        df["anos_no_programa"] = 0

    # Fill missing Pedra with "Desconhecido"
    for col in ("pedra_22", "pedra_21"):
        if col in df.columns:
            df[col] = df[col].fillna("Desconhecido")
        else:
            df[col] = "Desconhecido"

    # Fill missing gender
    df["genero"] = df["genero"].fillna("Desconhecido")

    # Fill subject grades with column median (few nulls)
    for col in ("matem", "portug"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Binary target: at risk = student not ahead of school grade
    if "defas" in df.columns:
        df[TARGET] = (df["defas"] >= 0).astype(int)

    return df


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
        ("encoder", OrdinalEncoder(
            categories=[
                PEDRA_ORDER,
                PEDRA_ORDER,
                ["Desconhecido", "Menino", "Menina"],
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformers = []

    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers)


def prepare_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: load → clean → return X, y."""
    df_raw = load_raw_data()
    df = clean_data(df_raw)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns: %s", missing)

    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = df[TARGET]

    logger.info(
        "Dataset ready — shape: %s | target distribution: %s",
        X.shape,
        y.value_counts().to_dict(),
    )
    return X, y
