import hopsworks
import joblib
import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Verbindung zu Hopsworks herstellen
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT", "mlops_project")
)
fs = project.get_feature_store()
mr = project.get_model_registry()

# 2. Modell aus der Registry herunterladen
# Holt die Version 1 des Modells
model_metadata = mr.get_model("volatility_model", version=1)
model_path = model_metadata.download()
model = joblib.load(os.path.join(model_path, "model.pkl"))

# 3. Aktuelle Daten abrufen (RT-Feature & Vorbereitung)
ticker = "BTC-USD"
# Wir holen die allerneuesten Daten für das RT-Feature
recent_data = yf.download(ticker, period="2d", interval="1d")
if isinstance(recent_data.columns, pd.MultiIndex):
    recent_data.columns = recent_data.columns.get_level_values(0)
recent_data.reset_index(inplace=True)

# RT-Feature: Aktuelle Day-Range (wird erst zum Inferenzzeitpunkt berechnet)
current_day_range = (recent_data['High'].iloc[-1] - recent_data['Low'].iloc[-1]) / recent_data['Low'].iloc[-1]

# 4. Aggregierte Features aus dem Feature Store holen
feature_view = fs.get_feature_view(name="stock_volatility_view", version=1)

# Wir holen die neuesten verfügbaren Features für diesen Ticker
# Hopsworks nutzt hier die Feature View, um den aktuellsten Stand zu liefern
batch_data = feature_view.get_batch_data()
latest_volatility_30d = batch_data['volatility_30d'].iloc[-1]

# 5. Prediction durchführen
# Wir bauen den Input-Vektor genau so auf, wie das Modell trainiert wurde
prediction_input = pd.DataFrame([{
    'volatility_30d': latest_volatility_30d,
    'day_range_pct': current_day_range
}])

prediction = model.predict(prediction_input)
probability = model.predict_proba(prediction_input)

# 6. Ergebnis ausgeben
status = "HOCH" if prediction[0] == 1 else "NIEDRIG"
print(f"\n--- Aktuelle Werte für {ticker} ---")
print(f"Volatilität (30-Tage Rolling):  {latest_volatility_30d:.4f}  ({latest_volatility_30d * 100:.2f}%)")
print(f"Tagesrange (High-Low):          {float(current_day_range):.4f}  ({float(current_day_range) * 100:.2f}%)")
print(f"\n--- Vorhersage für morgen ---")
print(f"Erwartete Volatilität:          {status}")
print(f"Wahrscheinlichkeit (hoch):      {probability[0][1] * 100:.2f}%")
print(f"Wahrscheinlichkeit (niedrig):   {probability[0][0] * 100:.2f}%")