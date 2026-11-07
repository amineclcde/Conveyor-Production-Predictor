# Conveyor Production Predictor

Application Flask de suivi de production de trois convoyeurs (Belt A/B/C) et de prédiction de production horaire par **régression linéaire**.

## Installation rapide

```bash
cd ConveyorProductionPredictor
python -m venv .venv
# Linux/macOS : source .venv/bin/activate
# Windows : .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Ouvrir `http://127.0.0.1:5000`, puis cliquer sur **Charger la démo**, et **Entraîner le modèle**. Le modèle est créé dans `model/model.pkl`.

## MySQL (option production)

1. Créer la base et les tables : `mysql -u root -p < database/schema.sql`.
2. Définir `DATABASE_URL`, par exemple :
   ```bash
   export DATABASE_URL='mysql+pymysql://root:motdepasse@localhost:3306/conveyor_predictor'
   ```
3. Lancer l'application. Sans cette variable, SQLite (`instance/conveyor.db`) est utilisé pour une démonstration sans configuration.

## Format CSV

Colonnes : `record_time, belt, conveyor_speed, active_trucks, humidity, temperature, operating_hours, shift_name, production`.

Les valeurs de `belt` attendues sont `Belt A`, `Belt B`, `Belt C`; les équipes peuvent être `Morning`, `Afternoon`, `Night`. Un exemple réaliste est inclus dans `data/production.csv`.

## Fonctions

- Import CSV, nettoyage (types, doublons, valeurs incohérentes), stockage SQL
- KPIs, tendances, production par convoyeur/équipe, nuage vitesse-production
- Entraînement sklearn avec encodage des variables catégorielles
- Prédiction et historique persistants

> Les données fournies sont synthétiques et doivent être remplacées par des mesures opérationnelles réelles avant toute décision industrielle.
