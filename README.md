# MLOps Projektarbeit – BTC Volatilitäts-Vorhersage

Dieses Projekt implementiert eine FTI-Architektur (Feature, Training, Inference) zur Vorhersage, ob die Volatilität von Bitcoin (BTC-USD) am nächsten Handelstag hoch sein wird (> 2%).

## Autoren

Diese Arbeit wurde von folgenden Personen erledigt:

* Silas Imboden
* Thajakan Thirunavukkarasu

## Daten & Modell

**Datenquelle:** [yfinance](https://github.com/ranaroussi/yfinance) – historische und aktuelle OHLCV-Daten für BTC-USD  
**Featurestore:** [Hopsworks](https://www.hopsworks.ai/)  
**Modell:** Random Forest Classifier (scikit-learn)  
**Target:** `target_volatility` – binär (1 = hohe Volatilität am nächsten Tag, 0 = niedrige)

### Features

| Feature | Typ | Beschreibung |
|---|---|---|
| `volatility_30d` | Aggregiert (Batch) | Rollende 30-Tage Standardabweichung der täglichen Rendite |
| `day_range_pct` | RT-Feature | Tagesrange in % `(High - Low) / Low` – erst zum Inferenzzeitpunkt bekannt |

Das **aggregierte Feature** `volatility_30d` wird in der Feature-Pipeline berechnet und im Hopsworks Featurestore gespeichert. Das **RT-Feature** `day_range_pct` wird in der Inference-Pipeline live von der yfinance API abgerufen und erst direkt bei der Vorhersage verwendet.

## Pipeline-Architektur

```
Feature-Pipeline          Training-Pipeline         Inference-Pipeline
─────────────────         ─────────────────         ──────────────────
yfinance API          →   Feature Store         →   Feature Store
  (historisch)              Feature View              (neueste Features)
Feature Engineering         train_test_split     +   yfinance API (live RT)
  volatility_30d            Random Forest             Model Registry
  day_range_pct             Model Registry       →   Prediction
Hopsworks Feature Store
```

### 1. Feature-Pipeline (`feature-pipeline.py`)
1. Historische BTC-Daten via yfinance abrufen (2023-01-01 bis 2026-04-20)
2. Aggregiertes Feature berechnen: 30-Tage Rolling-Volatilität
3. RT-Feature berechnen: tägliche High-Low-Range
4. Target-Label erstellen (shift(-1) für nächsten Tag)
5. Feature Group `stock_volatility_fg` in Hopsworks erstellen und befüllen

### 2. Training-Pipeline (`training-pipeline.py`)
1. Feature Group aus Hopsworks laden
2. Feature View `stock_volatility_view` erstellen
3. Training-/Testdaten via `train_test_split` aus der Feature View erstellen
4. Random Forest Classifier trainieren und evaluieren
5. Modell lokal sichern (joblib) und in Hopsworks Model Registry hochladen

### 3. Inference-Pipeline (`inference-pipeline.py`)
1. Modell `volatility_model` aus der Hopsworks Model Registry herunterladen
2. Neueste `volatility_30d` aus der Feature View holen
3. Aktuelles RT-Feature `day_range_pct` live von yfinance abrufen
4. Vorhersage durchführen und Ergebnis ausgeben

## Setup & Ausführen

### Voraussetzungen
- Python `< 3.14` (Hopsworks-Anforderung)
- Hopsworks-Account auf [hopsworks.ai](https://www.hopsworks.ai/) (kostenlos)

### Installation

```bash
# Virtuelle Umgebung erstellen und aktivieren
uv venv --python 3.12
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Abhängigkeiten installieren
uv pip install -r requirements.txt
```

### Konfiguration

`.env`-Datei im Projektverzeichnis erstellen:

```
HOPSWORKS_API_KEY=<dein_hopsworks_api_key>
HOPSWORKS_PROJECT=mlops_project
```

Den API-Key erhält man unter: Hopsworks → Account Settings → API Keys

> **Hinweis:** `HOPSWORKS_PROJECT` ist optional und gibt den Namen des Hopsworks-Projekts an. Wird die Variable nicht gesetzt, wird `mlops_project` als Standardwert verwendet.

### Pipelines ausführen

Die Pipelines müssen in dieser Reihenfolge ausgeführt werden:

```bash
# 1. Features berechnen und in Hopsworks speichern
python feature-pipeline.py

# 2. Modell trainieren und in Registry hochladen
python training-pipeline.py

# 3. Vorhersage für den nächsten Tag durchführen (CLI)
python inference-pipeline.py
```

## Web App (Inferenz im Browser)

Anstelle der CLI-Inferenz-Pipeline kann die Vorhersage auch über eine Webapplikation abgerufen werden (`app.py`). Die App zeigt die aktuellen Eingabewerte beider Features sowie die Vorhersage mit Wahrscheinlichkeiten an.

```bash
python app.py
```

Anschliessend im Browser öffnen: `http://localhost:5000`

> **Hinweis:** Feature-Pipeline und Training-Pipeline müssen vor dem Start der Web App ausgeführt worden sein, da die App auf das Modell in der Hopsworks Model Registry und die Feature View zugreift.

## Limitationen

- **Historische Daten:** Die Feature-Pipeline lädt Daten bis April 2026. Für eine kontinuierliche Nutzung müsste die Pipeline regelmässig (z.B. täglich) ausgeführt werden, um neue Daten nachzuladen.
- **RT-Feature bei Training:** `day_range_pct` wird beim Training mit historischen Werten verwendet. In einer vollständigen Produktionslösung würde dieses Feature über eine separate RT-Feature-Group und Point-in-Time-Joins korrekt abgebildet.
- **Kein automatisches Retraining:** Das Modell wird manuell durch Ausführen der Training-Pipeline aktualisiert.
