"""Model training pipeline.

Usage:
    python src/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import compare_models, evaluate_model, print_summary
from src.feature_engineering import create_features, get_extended_feature_columns
from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    clean_data,
    load_raw_data,
)
from src.utils import save_model, setup_logging

logger = setup_logging("train")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def _build_candidates() -> dict:
    """Return candidate classifier definitions (without preprocessor)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_model():
    """Full training pipeline: load → preprocess → compare → save best model."""

    # data
    logger.info("Loading and preparing data...")
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    df = create_features(df)

    numeric_feature_cols = get_extended_feature_columns(df)
    categorical_feature_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = numeric_feature_cols + categorical_feature_cols

    X = df[feature_cols]
    y = df[TARGET]

    logger.info("Class distribution — %s", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # cross-validate models
    preprocessor = build_preprocessor(
        numeric_features=numeric_feature_cols,
        categorical_features=categorical_feature_cols,
    )
    candidates = _build_candidates()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results: dict[str, float] = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_results[name] = scores.mean()
        logger.info("%s CV F1 = %.4f ± %.4f", name, scores.mean(), scores.std())

    best_name = max(cv_results, key=cv_results.get)
    logger.info("Best model by CV F1: %s", best_name)

    # train best on full train
    best_clf = candidates[best_name]
    final_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", best_clf),
    ])
    final_pipe.fit(X_train, y_train)

    # evaluate on test set
    test_results = {}
    for name, clf in candidates.items():
        p = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        p.fit(X_train, y_train)
        test_results[name] = evaluate_model(p, X_test, y_test)

    print("\n=== Test-set comparison ===")
    print_summary(test_results)
    print()

    best_test_name = compare_models(test_results)
    if best_test_name != best_name:
        logger.info(
            "Test-set best (%s) differs from CV best (%s); using CV best.",
            best_test_name,
            best_name,
        )

    best_metrics = test_results[best_name]
    print(f"\nSelected model: {best_name}")
    print(best_metrics["classification_report"])

    # serialize model
    metadata = {
        "model_name": best_name,
        "feature_columns": feature_cols,
        "trained_at": datetime.utcnow().isoformat(),
        "test_metrics": {
            k: v for k, v in best_metrics.items()
            if k != "classification_report"
        },
        "cv_f1": cv_results[best_name],
        "target": TARGET,
        "target_definition": "1 = student not ahead of school grade (Defas >= 0)",
    }
    save_model(final_pipe, metadata)
    logger.info("Training complete.")
    return final_pipe, metadata


if __name__ == "__main__":
    train_model()
