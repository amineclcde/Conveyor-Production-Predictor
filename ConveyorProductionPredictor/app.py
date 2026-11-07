"""Application Flask — Conveyor Belt Production Predictor.

Lancement depuis le dossier principal :
    python app.py

La base MySQL est utilisée lorsque DATABASE_URL est défini dans .env.
Sans configuration MySQL valide, l'application utilise SQLite automatiquement.
"""
from pathlib import Path
import os

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from sqlalchemy import create_engine, text

from model.predict import model_metrics, predict
from model.train import MODEL_PATH, train_model

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
SQLITE_URL = f"sqlite:///{ROOT / 'instance' / 'conveyor.db'}"

REQUIRED_COLUMNS = [
    "record_time",
    "belt",
    "conveyor_speed",
    "active_trucks",
    "humidity",
    "temperature",
    "operating_hours",
    "shift_name",
    "production",
]

# Accepte quelques noms fréquents lors de l'import de fichiers CSV.
COLUMN_ALIASES = {
    "date": "record_time",
    "datetime": "record_time",
    "timestamp": "record_time",
    "conveyor speed": "conveyor_speed",
    "speed": "conveyor_speed",
    "active trucks": "active_trucks",
    "trucks": "active_trucks",
    "operating hours": "operating_hours",
    "hours": "operating_hours",
    "shift": "shift_name",
    "production (tons/hour)": "production",
    "production tons/hour": "production",
}


def load_environment() -> None:
    """Charge les variables du fichier .env, s'il existe.

    Le fichier est volontairement lu sans dépendance externe. Les variables déjà
    définies dans Windows/macOS/Linux restent prioritaires.
    """
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_database_engine():
    """Crée l'accès MySQL ou bascule vers SQLite en cas de problème."""
    load_environment()
    database_url = os.getenv("DATABASE_URL", "").strip()

    if database_url:
        try:
            mysql_engine = create_engine(database_url, future=True, pool_pre_ping=True)
            with mysql_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            print("✓ Base de données active : MySQL")
            return mysql_engine, "MySQL"
        except Exception as error:
            print(f"! Connexion MySQL impossible ({error.__class__.__name__}). SQLite sera utilisée.")

    sqlite_engine = create_engine(SQLITE_URL, future=True)
    print("✓ Base de données active : SQLite (mode démonstration)")
    return sqlite_engine, "SQLite"


engine, DATABASE_TYPE = build_database_engine()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-development-secret")


def initialise_database() -> None:
    """Crée les tables si elles n'existent pas déjà."""
    identifier = (
        "INTEGER PRIMARY KEY AUTOINCREMENT"
        if engine.dialect.name == "sqlite"
        else "INT UNSIGNED AUTO_INCREMENT PRIMARY KEY"
    )

    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS production_records (
                id {identifier},
                record_time DATETIME NOT NULL,
                belt VARCHAR(80) NOT NULL,
                conveyor_speed FLOAT NOT NULL,
                active_trucks INTEGER NOT NULL,
                humidity FLOAT NOT NULL,
                temperature FLOAT NOT NULL,
                operating_hours FLOAT NOT NULL,
                shift_name VARCHAR(30) NOT NULL,
                production FLOAT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id {identifier},
                predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                belt VARCHAR(80) NOT NULL,
                conveyor_speed FLOAT NOT NULL,
                active_trucks INTEGER NOT NULL,
                humidity FLOAT NOT NULL,
                temperature FLOAT NOT NULL,
                operating_hours FLOAT NOT NULL,
                shift_name VARCHAR(30) NOT NULL,
                prediction FLOAT NOT NULL
            )
        """))


def get_production_records() -> pd.DataFrame:
    """Retourne les relevés de production ordonnés par date."""
    return pd.read_sql("SELECT * FROM production_records ORDER BY record_time", engine)


def clean_imported_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Normalise et valide les données avant insertion en base."""
    raw_data.columns = [
        COLUMN_ALIASES.get(
            str(column).strip().lower(),
            str(column).strip().lower().replace(" ", "_"),
        )
        for column in raw_data.columns
    ]

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw_data.columns))
    if missing_columns:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing_columns))

    data = raw_data[REQUIRED_COLUMNS].copy()
    data["record_time"] = pd.to_datetime(data["record_time"], errors="coerce")

    numeric_columns = [
        "conveyor_speed",
        "active_trucks",
        "humidity",
        "temperature",
        "operating_hours",
        "production",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["belt"] = data["belt"].fillna("Belt A").astype(str).str.strip()
    data["shift_name"] = data["shift_name"].fillna("Morning").astype(str).str.strip()
    data = data.dropna().drop_duplicates()

    valid_rows = (
        (data["conveyor_speed"] > 0)
        & (data["active_trucks"] >= 0)
        & (data["humidity"].between(0, 100))
        & (data["operating_hours"] > 0)
        & (data["production"] >= 0)
    )
    return data.loc[valid_rows].copy()


@app.route("/")
def home():
    data = get_production_records()
    if data.empty:
        return render_template("home.html", empty=True, database_type=DATABASE_TYPE)

    kpis = {
        "avg": data["production"].mean(),
        "max": data["production"].max(),
        "min": data["production"].min(),
        "speed": data["conveyor_speed"].mean(),
        "count": len(data),
    }
    production_by_belt = data.groupby("belt")["production"].sum().round(1)
    production_by_shift = data.groupby("shift_name")["production"].mean().round(1)
    daily_production = (
        data.assign(day=pd.to_datetime(data["record_time"]).dt.strftime("%Y-%m-%d"))
        .groupby("day")["production"]
        .mean()
        .tail(14)
    )
    scatter_data = data.sample(min(150, len(data)), random_state=1)

    metrics = model_metrics() if MODEL_PATH.exists() else None
    return render_template(
        "home.html",
        empty=False,
        database_type=DATABASE_TYPE,
        k=kpis,
        belt_labels=list(production_by_belt.index),
        belt_values=list(production_by_belt.values),
        shift_labels=list(production_by_shift.index),
        shift_values=list(production_by_shift.values),
        days=list(daily_production.index),
        daily=list(daily_production.values),
        scatter=scatter_data.to_dict("records"),
        metrics=metrics,
    )


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        flash("Veuillez sélectionner un fichier CSV.", "error")
        return redirect(url_for("home"))

    try:
        clean_data = clean_imported_data(pd.read_csv(uploaded_file))
        if clean_data.empty:
            raise ValueError("Aucune ligne valide après le nettoyage.")
        clean_data.to_sql("production_records", engine, if_exists="append", index=False)
        flash(f"{len(clean_data)} lignes propres importées avec succès.", "success")
    except Exception as error:
        flash(f"Import impossible : {error}", "error")
    return redirect(url_for("home"))


@app.route("/seed", methods=["POST"])
def seed_data():
    """Charge le jeu de données synthétique seulement si la table est vide."""
    if not get_production_records().empty:
        flash("La base contient déjà des données.", "error")
        return redirect(url_for("home"))

    try:
        sample_data = clean_imported_data(pd.read_csv(ROOT / "data" / "production.csv"))
        sample_data.to_sql("production_records", engine, if_exists="append", index=False)
        flash(f"{len(sample_data)} observations de démonstration ajoutées.", "success")
    except Exception as error:
        flash(f"Chargement impossible : {error}", "error")
    return redirect(url_for("home"))


@app.route("/train", methods=["POST"])
def train():
    data = get_production_records()
    if len(data) < 10:
        flash("Au moins 10 lignes sont nécessaires pour entraîner le modèle.", "error")
    else:
        try:
            metrics = train_model(data, DATABASE_TYPE)
            flash(
                f"Modèle entraîné : MAE {metrics['mae']:.2f} t/h, R² {metrics['r2']:.3f}.",
                "success",
            )
        except Exception as error:
            flash(f"Entraînement impossible : {error}", "error")
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
def prediction():
    result = None
    if request.method == "POST":
        if not MODEL_PATH.exists():
            flash("Entraînez d’abord le modèle.", "error")
            return redirect(url_for("home"))

        try:
            values = {
                "belt": request.form["belt"],
                "conveyor_speed": float(request.form["conveyor_speed"]),
                "active_trucks": int(request.form["active_trucks"]),
                "humidity": float(request.form["humidity"]),
                "temperature": float(request.form["temperature"]),
                "operating_hours": float(request.form["operating_hours"]),
                "shift_name": request.form["shift_name"],
            }
            result = predict(values)
            pd.DataFrame([{**values, "prediction": result}]).to_sql(
                "prediction_history", engine, if_exists="append", index=False
            )
        except Exception as error:
            flash(f"Valeurs invalides : {error}", "error")

    history = pd.read_sql(
        "SELECT * FROM prediction_history ORDER BY predicted_at DESC LIMIT 12",
        engine,
    )
    return render_template(
        "predict.html",
        result=result,
        history=history.to_dict("records"),
        database_type=DATABASE_TYPE,
    )


initialise_database()

if __name__ == "__main__":
    app.run(debug=True)
