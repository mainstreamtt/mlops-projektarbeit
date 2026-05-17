import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
import joblib
import os

# Load .env file
load_dotenv()

# 1. Verbindung zu Hopsworks & Featurestore
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project="mlops_project"
)
fs = project.get_feature_store()

# 2. Feature Group abrufen
volatility_fg = fs.get_feature_group(name="stock_volatility_fg", version=1)

# 3. Feature View erstellen (oder laden, falls schon vorhanden)
try:
    # Nur numerische Features auswählen (event_time ist auch nicht für Training geeignet)
    query = volatility_fg.select(["volatility_30d", "day_range_pct", "target_volatility"])
    feature_view = fs.get_or_create_feature_view(
        name="stock_volatility_view",
        version=3,
        labels=["target_volatility"], # Hier definieren wir das Target
        query=query
    )
except Exception as e:
    print(f"Fehler beim Erstellen der Feature View: {e}")
    feature_view = fs.get_feature_view(name="stock_volatility_view", version=3)

# 4. Training Dataset erstellen
# Splittet die Daten intern in Training und Test
X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)

# String-Spalten entfernen (ticker ist ein String und nicht für Training geeignet)
for df in [X_train, X_test]:
    if 'ticker' in df.columns:
        df.drop(columns=['ticker'], inplace=True)

# 5. Modell trainieren
# Modell-Performance ist laut Aufgabe nicht wichtig
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train.values.ravel())

# 6. Modell-Evaluation (optional, aber gut für die Doku)
accuracy = model.score(X_test, y_test)
print(f"Modell trainiert mit Accuracy: {accuracy}")

# 7. Modell lokal speichern und in Registry hochladen
model_dir = "stock_model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Lokal sichern 
joblib.dump(model, model_dir + "/model.pkl")

# In Hopsworks Model Registry hochladen
mr = project.get_model_registry()
stock_model = mr.python.create_model(
    name="volatility_model",
    metrics={"accuracy": accuracy},
    description="Random Forest Modell zur Vorhersage von Aktien-Volatilität"
)
stock_model.save(model_dir)
print("Modell erfolgreich in die Registry hochgeladen!")