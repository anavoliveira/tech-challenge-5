"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload

Run directly:
    python app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.route import router
from src.utils import setup_logging

logger = setup_logging("api")

app = FastAPI(
    title="Passos Magicos — Previsao de Risco de Defasagem Escolar",
    description=(
        "API REST para predicao de risco de defasagem escolar de alunos da "
        "Associacao Passos Magicos, baseada em indicadores educacionais."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("API startup — loading model...")
    try:
        from app.route import _get_pipeline
        _get_pipeline()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning("Model not loaded at startup: %s", exc)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
