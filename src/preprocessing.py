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


# pedra vai de ametista (pior) até topázio (melhor)
PEDRA_ORDER = ["Desconhecido", "Ametista", "Ágata", "Quartzo", "Topázio"]

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

# mapeamento das colunas do excel para os nomes internos
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

# a coluna de gênero vem com encoding diferente dependendo da versão do excel
_GENERO_RAW_OPTIONS = ["Gênero", "Genero", "gênero", "genero"]


def load_raw_data() -> pd.DataFrame:
    path = get_database_path() / "base_2024.xlsx"
    df = pd.read_excel(path)
    logger.info("carregou %s: %s linhas", path.name, df.shape[0])
    return df


def _resolve_genero_column(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia a coluna de gênero independente do encoding."""
    for candidate in _GENERO_RAW_OPTIONS:
        if candidate in df.columns:
            return df.rename(columns={candidate: "genero"})
    # tenta match parcial caso venha com alguma variação estranha
    for col in df.columns:
        if col.lower().startswith("g") and "nero" in col.lower():
            return df.rename(columns={col: "genero"})
    logger.warning("coluna de gênero não encontrada, criando dummy")
    df["genero"] = "Desconhecido"
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns=_COLUMN_MAP)
    df = _resolve_genero_column(df)

    # anos no programa = ano atual (2022) - ingresso + 1
    if "ano_ingresso" in df.columns:
        df["anos_no_programa"] = 2022 - df["ano_ingresso"] + 1
    else:
        df["anos_no_programa"] = 0

    for col in ("pedra_22", "pedra_21"):
        if col in df.columns:
            df[col] = df[col].fillna("Desconhecido")
        else:
            df[col] = "Desconhecido"

    df["genero"] = df["genero"].fillna("Desconhecido")

    # poucos nulos em notas, preenche com mediana
    for col in ("matem", "portug"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # target: risco = 1 quando aluno NÃO está adiantado (defas >= 0)
    # importante: IAN foi excluído pois é discretização direta de defas (seria data leakage)
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
    df_raw = load_raw_data()
    df = clean_data(df_raw)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        logger.warning("colunas ausentes: %s", missing)

    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = df[TARGET]

    logger.info("dataset pronto — shape: %s | distribuição target: %s", X.shape, y.value_counts().to_dict())
    return X, y
