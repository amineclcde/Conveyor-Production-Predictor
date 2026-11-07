"""Entraînement du modèle de prédiction de production des convoyeurs.

Exécution depuis VS Code ou un terminal :
    python model/train.py

Priorité des données :
1. Table MySQL production_records (si DATABASE_URL est configurée et contient des données)
2. Fichier local data/production.csv (solution de secours)
"""
from pathlib import Path
from datetime import datetime
import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Permet l'exécution directe : python model/train.py depuis n'importe quel dossier.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "data" / "production.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"
ENV_PATH = ROOT / ".env"
FEATURES = [
    "belt", "conveyor_speed", "active_trucks", "humidity", "temperature",
    "operating_hours", "shift_name",
]
TARGET = "production"


def load_environment() -> None:
    """Charge .env s'il existe, sans écraser les variables déjà définies."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_training_data() -> tuple[pd.DataFrame, str]:
    """Lit MySQL en priorité ; revient au CSV si MySQL n'est pas prêt."""
    load_environment()
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        try:
            from sqlalchemy import create_engine
            engine = create_engine(database_url, future=True)
            query = "SELECT record_time, belt, conveyor_speed, active_trucks, humidity, temperature, operating_hours, shift_name, production FROM production_records"
            data = pd.read_sql(query, engine)
            if len(data) >= 10:
                print(f"✓ Source : MySQL — {len(data)} enregistrements lus.")
                return data, "MySQL"
            print("! MySQL est accessible mais contient moins de 10 lignes ; utilisation du CSV.")
        except Exception as error:
            print(f"! MySQL indisponible ({error.__class__.__name__}) ; utilisation du CSV.")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")
    data = pd.read_csv(CSV_PATH)
    print(f"✓ Source : CSV — {len(data)} enregistrements lus depuis {CSV_PATH.name}.")
    return data, "CSV"


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Vérifie les colonnes minimales et retire les lignes non exploitables."""
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing))

    cleaned = data.copy()
    for column in ["conveyor_speed", "active_trucks", "humidity", "temperature", "operating_hours", TARGET]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=[TARGET])
    cleaned = cleaned[cleaned[TARGET] >= 0]
    if len(cleaned) < 10:
        raise ValueError("Au moins 10 lignes de production valides sont nécessaires.")
    return cleaned


def train_model(data: pd.DataFrame, source: str = "inconnue") -> dict:
    """Entraîne et sauvegarde la pipeline complète de régression linéaire."""
    data = validate_data(data)
    numeric_features = ["conveyor_speed", "active_trucks", "humidity", "temperature", "operating_hours"]
    categorical_features = ["belt", "shift_name"]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ])
    pipeline.fit(data[FEATURES], data[TARGET])
    estimated = pipeline.predict(data[FEATURES])

    metrics = {
        "records": int(len(data)),
        "mae": float(mean_absolute_error(data[TARGET], estimated)),
        "rmse": float(mean_squared_error(data[TARGET], estimated) ** 0.5),
        "r2": float(r2_score(data[TARGET], estimated)),
        "source": source,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    return metrics


def main() -> None:
    print("=" * 58)
    print("CONVEYOR BELT PRODUCTION PREDICTOR — ENTRAÎNEMENT")
    print("=" * 58)
    try:
        data, source = load_training_data()
        metrics = train_model(data, source)
        print("\n✓ Modèle entraîné et sauvegardé avec succès.")
        print(f"  Fichier : {MODEL_PATH}")
        print(f"  Source  : {metrics['source']}")
        print(f"  Lignes  : {metrics['records']}")
        print(f"  MAE     : {metrics['mae']:.2f} t/h")
        print(f"  RMSE    : {metrics['rmse']:.2f} t/h")
        print(f"  R²      : {metrics['r2']:.3f}")
    except Exception as error:
        print(f"\n✗ Erreur d’entraînement : {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
