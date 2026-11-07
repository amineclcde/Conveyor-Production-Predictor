"""Prédiction en ligne de commande.

Exécution depuis VS Code ou un terminal :
    python model/predict.py

Le même modèle est aussi utilisé automatiquement par l'application Flask.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.pkl"
FEATURES = [
    "belt", "conveyor_speed", "active_trucks", "humidity", "temperature",
    "operating_hours", "shift_name",
]


def load_saved_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Aucun modèle trouvé. Exécutez d'abord : python model/train.py"
        )
    return joblib.load(MODEL_PATH)


def predict(values: dict) -> float:
    """Retourne la production estimée en tonnes par heure."""
    saved_model = load_saved_model()
    missing = [feature for feature in saved_model["features"] if feature not in values]
    if missing:
        raise ValueError("Paramètres manquants : " + ", ".join(missing))
    frame = pd.DataFrame([values])[saved_model["features"]]
    return float(saved_model["pipeline"].predict(frame)[0])


def model_metrics() -> dict:
    """Retourne les métriques affichables par Flask."""
    return load_saved_model().get("metrics", {})


def ask_text(label: str, default: str) -> str:
    answer = input(f"{label} [{default}] : ").strip()
    return answer if answer else default


def ask_number(label: str, default: float, kind=float):
    while True:
        answer = input(f"{label} [{default}] : ").strip()
        try:
            return kind(answer) if answer else kind(default)
        except ValueError:
            print("Veuillez entrer une valeur numérique valide.")


def main() -> None:
    print("=" * 58)
    print("CONVEYOR BELT PRODUCTION PREDICTOR — PRÉDICTION")
    print("=" * 58)
    try:
        values = {
            "belt": ask_text("Convoyeur (Belt A / Belt B / Belt C)", "Belt A"),
            "conveyor_speed": ask_number("Vitesse du convoyeur en m/s", 2.6),
            "active_trucks": ask_number("Nombre de camions actifs", 5, int),
            "humidity": ask_number("Humidité en %", 34.0),
            "temperature": ask_number("Température en °C", 28.0),
            "operating_hours": ask_number("Heures de fonctionnement", 8.0),
            "shift_name": ask_text("Équipe (Morning / Afternoon / Night)", "Morning"),
        }
        result = predict(values)
        print("\n" + "─" * 58)
        print(f"PRODUCTION ATTENDUE : {result:.1f} tonnes/heure")
        print("─" * 58)
        metrics = model_metrics()
        if metrics:
            print(f"Modèle entraîné le : {metrics.get('trained_at', 'non renseigné')}")
            print(f"Source des données : {metrics.get('source', 'non renseignée')}")
    except Exception as error:
        print(f"\n✗ Erreur de prédiction : {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
