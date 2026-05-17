from dotenv import load_dotenv
import os

# Load .env file (typically at the top of your main file)
load_dotenv()

# Access variables
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

import yfinance as yf
import pandas as pd
import hopsworks
import datetime

# 1. Verbindung zu Hopsworks herstellen [cite: 12]
project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project="mlops_project",
    )
fs = project.get_feature_store()

# 2. Rohdaten via API abrufen [cite: 37]
# Beispiel: Apple Aktie (AAPL)
ticker = "BTC-USD"
data = yf.download(ticker, start="2023-01-01", end="2026-04-20")

# Flatten MultiIndex columns from yfinance (if present)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

# 3. Feature Engineering [cite: 28, 38]

# A) Aggregiertes Feature: 30-Tage Volatilität (Standardabweichung der Rendite) [cite: 31]
data['returns'] = data['Close'].pct_change()
data['volatility_30d'] = data['returns'].rolling(window=30).std()

# B) Aktuelles (RT) Feature: Intraday-Range (High-Low Differenz) [cite: 32]
data['day_range_pct'] = (data['High'] - data['Low']) / data['Low']

# C) Target: Wird die Volatilität morgen hoch sein (> 2%)? [cite: 10, 39]
# Wir verschieben die Volatilität um -1, um das "Label" für den nächsten Tag zu erhalten
data['target_volatility'] = (data['volatility_30d'].shift(-1) > 0.02).astype(int)

# Bereinigung: NaNs durch Rolling Windows und Shifting entfernen
data.dropna(inplace=True)

# 4. Datenstruktur für Hopsworks vorbereiten 
# Hopsworks benötigt einen Primary Key und eine Event Time
data['ticker'] = ticker
data['event_time'] = data['Date'].apply(lambda x: int(x.timestamp() * 1000)) # In Millisekunden

# Nur die relevanten Spalten behalten [cite: 39]
df_features = data[['ticker', 'event_time', 'volatility_30d', 'day_range_pct', 'target_volatility']]

# Sicherstellen, dass alle Spaltennamen lowercase sind (Hopsworks Anforderung)
df_features.columns = [col.lower() for col in df_features.columns]

# 5. Feature Group erstellen und Daten einfügen [cite: 40, 41]
volatility_fg = fs.get_or_create_feature_group(
    name="stock_volatility_fg",
    version=1,
    primary_key=['ticker'],
    event_time='event_time',
    description="Aktien-Volatilitäts-Features (Aggregiert & RT)"
)

volatility_fg.insert(df_features)
print("Feature Pipeline erfolgreich abgeschlossen und Daten in Hopsworks gespeichert!")