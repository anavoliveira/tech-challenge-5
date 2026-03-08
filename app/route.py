import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering import create_features
from src.preprocessing import ALL_FEATURES, clean_data
from src.utils import load_model, setup_logging

logger = setup_logging("api.route")
router = APIRouter()


class StudentFeatures(BaseModel):
    """
    Dados de entrada de um aluno para predição.

    Nota: IAN não é incluído pois é derivado diretamente de defas
    (seria data leakage se usado como feature).
    """

    fase: int = Field(..., ge=0, le=8)
    idade: int = Field(..., ge=5, le=30)
    ano_ingresso: int = Field(..., ge=2010, le=2024)
    inde: float = Field(..., ge=0.0, le=10.0)
    iaa: float = Field(..., ge=0.0, le=10.0)
    ieg: float = Field(..., ge=0.0, le=10.0)
    ips: float = Field(..., ge=0.0, le=10.0)
    ida: float = Field(..., ge=0.0, le=10.0)
    ipv: float = Field(..., ge=0.0, le=10.0)
    cg: float = Field(..., ge=0.0)
    cf: float = Field(..., ge=0.0)
    ct: float = Field(..., ge=0.0)
    matem: float = Field(..., ge=0.0, le=10.0)
    portug: float = Field(..., ge=0.0, le=10.0)
    pedra_22: str = Field("Ametista")
    pedra_21: Optional[str] = Field(None)
    genero: str = Field("Desconhecido")

    class Config:
        json_schema_extra = {
            "example": {
                "fase": 3,
                "idade": 12,
                "ano_ingresso": 2018,
                "inde": 6.5,
                "iaa": 7.0,
                "ieg": 5.5,
                "ips": 6.0,
                "ida": 6.2,
                "ipv": 7.5,
                "cg": 300,
                "cf": 10,
                "ct": 8,
                "matem": 6.5,
                "portug": 6.8,
                "pedra_22": "Ametista",
                "pedra_21": "Ametista",
                "genero": "Menino",
            }
        }


class PredictionResponse(BaseModel):
    risco_defasagem: float
    classificacao: str
    confianca: float
    modelo: str


_pipeline = None
_metadata: dict = {}


def _get_pipeline():
    global _pipeline, _metadata
    if _pipeline is None:
        try:
            _pipeline, _metadata = load_model()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Modelo não encontrado. Execute 'python src/train.py' primeiro.",
            ) from exc
    return _pipeline, _metadata


def _classify(prob: float) -> str:
    if prob < 0.35:
        return "Baixo Risco"
    if prob < 0.65:
        return "Medio Risco"
    return "Alto Risco"


def _features_from_input(data: StudentFeatures) -> pd.DataFrame:
    row = {
        "fase": data.fase,
        "idade": data.idade,
        "ano_ingresso": data.ano_ingresso,
        # anos_no_programa precisa ser calculado aqui pois clean_data não é chamado na API
        "anos_no_programa": 2022 - data.ano_ingresso + 1,
        "inde": data.inde,
        "iaa": data.iaa,
        "ieg": data.ieg,
        "ips": data.ips,
        "ida": data.ida,
        "ipv": data.ipv,
        "cg": data.cg,
        "cf": data.cf,
        "ct": data.ct,
        "matem": data.matem,
        "portug": data.portug,
        "pedra_22": data.pedra_22,
        "pedra_21": data.pedra_21 if data.pedra_21 else "Desconhecido",
        "genero": data.genero,
        "defas": 0,
    }

    df = pd.DataFrame([row])
    df = create_features(df)

    _, metadata = _get_pipeline()
    feature_cols = metadata.get("feature_columns", [])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[feature_cols]


@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


@router.get("/ping", tags=["SageMaker"])
def sagemaker_ping():
    """SageMaker exige esse endpoint retornando 200."""
    return {"status": "ok"}


@router.post("/invocations", response_model=PredictionResponse, tags=["SageMaker"])
def sagemaker_invocations(student: StudentFeatures):
    return predict(student)


@router.get("/model-info", tags=["Model"])
def model_info():
    _, meta = _get_pipeline()
    return meta


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentFeatures):
    pipeline, meta = _get_pipeline()

    try:
        X = _features_from_input(student)
        prob = float(pipeline.predict_proba(X)[0, 1])
    except Exception as exc:
        logger.error("erro na predição: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = PredictionResponse(
        risco_defasagem=round(prob, 4),
        classificacao=_classify(prob),
        confianca=round(max(prob, 1 - prob), 4),
        modelo=meta.get("model_name", "unknown"),
    )

    logger.info("predição: aluno=%s risco=%.4f classe=%s", student.model_dump(), prob, response.classificacao)

    return response
