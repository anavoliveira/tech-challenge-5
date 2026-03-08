import logging
import sys
from pathlib import Path

import joblib


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_database_path() -> Path:
    return get_project_root() / "database"


def get_model_path() -> Path:
    return get_project_root() / "app" / "model"


def setup_logging(name: str = "passos_magicos") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def save_model(pipeline, metadata: dict = None, filename: str = "model.joblib") -> Path:
    model_dir = get_model_path()
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / filename
    payload = {"pipeline": pipeline, "metadata": metadata or {}}
    joblib.dump(payload, dest)
    logger = setup_logging()
    logger.info("modelo salvo em %s", dest)
    return dest


def load_model(filename: str = "model.joblib"):
    dest = get_model_path() / filename
    if not dest.exists():
        raise FileNotFoundError(f"arquivo do modelo não encontrado: {dest}")
    payload = joblib.load(dest)
    return payload["pipeline"], payload.get("metadata", {})
