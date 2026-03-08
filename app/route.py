"""API route definitions."""

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

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class StudentFeatures(BaseModel):
    """Input features for a single student prediction.

    Note: IAN (Indice de Adequacao ao Nivel) is intentionally excluded
    because it is a direct discretization of the defasagem target and
    would constitute data leakage.
    """

    fase: int = Field(..., ge=0, le=8, description="Fase atual no programa (0-8)")
    idade: int = Field(..., ge=5, le=30, description="Idade do aluno")
    ano_ingresso: int = Field(..., ge=2010, le=2024, description="Ano de ingresso no programa")
    inde: float = Field(..., ge=0.0, le=10.0, description="Indice de Desenvolvimento Educacional")
    iaa: float = Field(..., ge=0.0, le=10.0, description="Indice de Auto Avaliacao")
    ieg: float = Field(..., ge=0.0, le=10.0, description="Indice de Engajamento")
    ips: float = Field(..., ge=0.0, le=10.0, description="Indice Psicossocial")
    ida: float = Field(..., ge=0.0, le=10.0, description="Indice de Desenvolvimento do Aprendizado")
    ipv: float = Field(..., ge=0.0, le=10.0, description="Indice do Ponto de Virada")
    cg: float = Field(..., ge=0.0, description="Conceito Geral")
    cf: float = Field(..., ge=0.0, description="Conceito Final")
    ct: float = Field(..., ge=0.0, description="Conceito Total")
    matem: float = Field(..., ge=0.0, le=10.0, description="Nota de Matematica")
    portug: float = Field(..., ge=0.0, le=10.0, description="Nota de Portugues")
    pedra_22: str = Field("Ametista", description="Classificacao Pedra 2022 (Ametista/Agata/Quartzo/Topazio)")
    pedra_21: Optional[str] = Field(None, description="Classificacao Pedra 2021 (opcional)")
    genero: str = Field("Desconhecido", description="Genero (Menino / Menina)")

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
    risco_defasagem: float = Field(..., description="Probabilidade de risco de defasagem (0-1)")
    classificacao: str = Field(..., description="Baixo Risco / Medio Risco / Alto Risco")
    confianca: float = Field(..., description="Confianca da predicao (0-1)")
    modelo: str = Field(..., description="Nome do modelo utilizado")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
                detail="Model not found. Run 'python src/train.py' first.",
            ) from exc
    return _pipeline, _metadata


def _classify(prob: float) -> str:
    if prob < 0.35:
        return "Baixo Risco"
    if prob < 0.65:
        return "Medio Risco"
    return "Alto Risco"


def _features_from_input(data: StudentFeatures) -> pd.DataFrame:
    """Convert API input into the feature DataFrame expected by the pipeline."""
    row = {
        "fase": data.fase,
        "idade": data.idade,
        "ano_ingresso": data.ano_ingresso,
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
        # dummy target (not used for prediction)
        "defas": 0,
    }
    df = pd.DataFrame([row])
    df = create_features(df)
    feature_cols = _metadata.get("feature_columns", ALL_FEATURES)
    available = [c for c in feature_cols if c in df.columns]
    return df[available]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", tags=["Health"])
def health_check():
    """Liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# SageMaker-required routes
# GET /ping        → health check (SageMaker expects 200)
# POST /invocations → predictions (SageMaker sends requests here)
# ---------------------------------------------------------------------------

@router.get("/ping", tags=["SageMaker"])
def sagemaker_ping():
    """SageMaker health check — must return 200."""
    return {"status": "ok"}


@router.post("/invocations", response_model=PredictionResponse, tags=["SageMaker"])
def sagemaker_invocations(student: StudentFeatures):
    """SageMaker inference endpoint — mirrors /predict."""
    return predict(student)


@router.get("/model-info", tags=["Model"])
def model_info():
    """Return metadata of the currently loaded model."""
    _, meta = _get_pipeline()
    return meta


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentFeatures):
    """Predict defasagem risk for a single student."""
    pipeline, meta = _get_pipeline()

    try:
        X = _features_from_input(student)
        prob = float(pipeline.predict_proba(X)[0, 1])
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = PredictionResponse(
        risco_defasagem=round(prob, 4),
        classificacao=_classify(prob),
        confianca=round(max(prob, 1 - prob), 4),
        modelo=meta.get("model_name", "unknown"),
    )

    logger.info(
        "prediction event=%s timestamp=%s input=%s output=%s model=%s",
        "prediction",
        pd.Timestamp.utcnow().isoformat(),
        student.model_dump(),
        {"risco_defasagem": response.risco_defasagem, "classificacao": response.classificacao},
        response.modelo,
    )

    return response
