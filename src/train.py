"""
Treina os modelos e salva o melhor.

Uso: python src/train.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import compare_models, evaluate_model, print_summary
from src.feature_engineering import create_features, get_extended_feature_columns
from src.preprocessing import (
    CATEGORICAL_FEATURES,
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
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
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
    logger.info("carregando dados...")
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    df = create_features(df)

    numeric_feature_cols = get_extended_feature_columns(df)
    categorical_feature_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = numeric_feature_cols + categorical_feature_cols

    X = df[feature_cols]
    y = df[TARGET]

    print(f"distribuição das classes: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info("treino: %d amostras | teste: %d amostras", len(X_train), len(X_test))

    preprocessor = build_preprocessor(
        numeric_features=numeric_feature_cols,
        categorical_features=categorical_feature_cols,
    )

    candidates = _build_candidates()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # cross-validation pra escolher o melhor modelo
    cv_results: dict[str, float] = {}
    for name, clf in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_results[name] = scores.mean()
        logger.info("%s → CV F1 = %.4f (±%.4f)", name, scores.mean(), scores.std())

    best_name = max(cv_results, key=cv_results.get)
    logger.info("melhor modelo no CV: %s", best_name)

    # avaliação no conjunto de teste pra confirmar
    test_results = {}
    for name, clf in candidates.items():
        p = Pipeline([
            ("preprocessor", build_preprocessor(
                numeric_features=numeric_feature_cols,
                categorical_features=categorical_feature_cols,
            )),
            ("classifier", clf),
        ])
        p.fit(X_train, y_train)
        test_results[name] = evaluate_model(p, X_test, y_test)

    print("\n=== Resultados no conjunto de teste ===")
    print_summary(test_results)
    print()

    best_test_name = compare_models(test_results)
    if best_test_name != best_name:
        # pode acontecer quando os scores são muito próximos
        logger.info("melhor no teste (%s) difere do CV (%s) — usando CV", best_test_name, best_name)

    best_metrics = test_results[best_name]
    print(f"\nModelo selecionado: {best_name}")
    print(best_metrics["classification_report"])

    # treina o pipeline final com todos os dados de treino
    final_pipe = Pipeline([
        ("preprocessor", build_preprocessor(
            numeric_features=numeric_feature_cols,
            categorical_features=categorical_feature_cols,
        )),
        ("classifier", candidates[best_name]),
    ])
    final_pipe.fit(X_train, y_train)

    metadata = {
        "model_name": best_name,
        "feature_columns": feature_cols,
        "trained_at": datetime.utcnow().isoformat(),
        "test_metrics": {k: v for k, v in best_metrics.items() if k != "classification_report"},
        "cv_f1": cv_results[best_name],
        "target": TARGET,
        "target_definition": "1 = aluno não adiantado na série (Defas >= 0)",
    }

    save_model(final_pipe, metadata)
    logger.info("treinamento concluído")

    return final_pipe, metadata


if __name__ == "__main__":
    train_model()
