from flask import Flask, render_template_string, jsonify
import hopsworks
import joblib
import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

_model = None
_feature_view = None


def get_resources():
    global _model, _feature_view
    if _model is None:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT", "mlops_project")
        )
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        model_meta = mr.get_model("volatility_model", version=1)
        model_path = model_meta.download()
        _model = joblib.load(os.path.join(model_path, "model.pkl"))
        _feature_view = fs.get_feature_view(name="stock_volatility_view", version=1)
    return _model, _feature_view


HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>BTC Volatility Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d0d14;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container { max-width: 560px; width: 100%; padding: 1.5rem; }
        h1 { font-size: 1.4rem; color: #fff; margin-bottom: 0.2rem; }
        .subtitle { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }

        /* ── Loading screen ── */
        #loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1.25rem;
            padding: 3rem 0;
            transition: opacity 0.3s;
        }
        .spinner-ring {
            width: 56px;
            height: 56px;
            border: 3px solid #1e1e2e;
            border-top-color: #6272a4;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-status {
            color: #666;
            font-size: 0.88rem;
            min-height: 1.2em;
            transition: opacity 0.4s;
        }

        /* ── Content ── */
        #content {
            display: none;
            opacity: 0;
            transition: opacity 0.4s;
        }
        #content.visible { display: block; opacity: 1; }

        .card {
            background: #181824;
            border: 1px solid #252535;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .card-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #555;
            margin-bottom: 1rem;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #1e1e2e;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #999; font-size: 0.9rem; }
        .metric-tag {
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            margin-left: 0.4rem;
            vertical-align: middle;
        }
        .tag-batch { background: #1e3a5f; color: #5ba3e8; }
        .tag-rt    { background: #2d1e4f; color: #a07fe8; }
        .metric-value { font-weight: 600; font-size: 1rem; color: #fff; }

        .prediction-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #555;
            margin-bottom: 0.6rem;
        }
        .prediction-badge { font-size: 2.2rem; font-weight: 700; margin-bottom: 1.2rem; }
        .high { color: #ff5c5c; }
        .low  { color: #50fa7b; }

        .prob-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #999;
            margin-bottom: 0.3rem;
        }
        .prob-val { font-weight: 600; color: #fff; }
        .bar { background: #1e1e2e; border-radius: 6px; height: 7px; margin-bottom: 0.9rem; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 6px; width: 0; transition: width 0.6s ease; }
        .bar-fill-high { background: #ff5c5c; }
        .bar-fill-low  { background: #50fa7b; }

        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            background: #3b4a7a;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            cursor: pointer;
            margin-top: 0.5rem;
        }
        .btn:hover { background: #4a5a9a; }
        .btn:disabled { background: #252535; color: #555; cursor: default; }

        .error-card { background: #2a1515; border-color: #5a2020; }
        .error-msg  { color: #ff8080; font-size: 0.9rem; }

        .footer { text-align: center; color: #444; font-size: 0.78rem; margin-top: 1.25rem; }
    </style>
</head>
<body>
<div class="container">
    <h1>&#8383; BTC-USD Volatility Predictor</h1>
    <p class="subtitle" id="subtitle">Vorhersage für den nächsten Handelstag</p>

    <!-- Loading state -->
    <div id="loading">
        <div class="spinner-ring"></div>
        <p class="loading-status" id="status-text">Verbinde mit Hopsworks...</p>
    </div>

    <!-- Content (filled by JS) -->
    <div id="content">
        <!-- Values card -->
        <div class="card" id="values-card">
            <div class="card-title">Aktuelle Eingabewerte</div>
            <div class="metric">
                <span class="metric-label">
                    30-Tage Volatilität
                    <span class="metric-tag tag-batch">Batch</span>
                </span>
                <span class="metric-value" id="val-volatility">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">
                    Tagesrange High&ndash;Low
                    <span class="metric-tag tag-rt">RT</span>
                </span>
                <span class="metric-value" id="val-day-range">—</span>
            </div>
        </div>

        <!-- Prediction card -->
        <div class="card" id="prediction-card">
            <div class="prediction-label">Vorhersage morgen</div>
            <div class="prediction-badge" id="pred-badge">—</div>

            <div class="prob-row">
                <span>Hohe Volatilität</span>
                <span class="prob-val" id="prob-high-val">—</span>
            </div>
            <div class="bar"><div class="bar-fill bar-fill-high" id="bar-high"></div></div>

            <div class="prob-row">
                <span>Niedrige Volatilität</span>
                <span class="prob-val" id="prob-low-val">—</span>
            </div>
            <div class="bar"><div class="bar-fill bar-fill-low" id="bar-low"></div></div>
        </div>

        <!-- Error card (hidden by default) -->
        <div class="card error-card" id="error-card" style="display:none">
            <div class="card-title">Fehler</div>
            <p class="error-msg" id="error-msg"></p>
        </div>
    </div>

    <button class="btn" id="refresh-btn" onclick="loadPrediction()">Aktualisieren</button>
    <p class="footer">Datenquelle: Yahoo Finance &middot; Featurestore: Hopsworks &middot; Modell: Random Forest</p>
</div>

<script>
    const STATUS_MESSAGES = [
        "Verbinde mit Hopsworks...",
        "Lade Modell aus Registry...",
        "Rufe Live-Daten von Yahoo Finance ab...",
        "Berechne Vorhersage..."
    ];

    let statusTimer = null;

    function startStatusCycle() {
        let i = 0;
        const el = document.getElementById("status-text");
        el.textContent = STATUS_MESSAGES[0];
        statusTimer = setInterval(() => {
            i = (i + 1) % STATUS_MESSAGES.length;
            el.style.opacity = 0;
            setTimeout(() => {
                el.textContent = STATUS_MESSAGES[i];
                el.style.opacity = 1;
            }, 200);
        }, 2200);
    }

    function stopStatusCycle() {
        clearInterval(statusTimer);
    }

    function showLoading() {
        document.getElementById("loading").style.display = "flex";
        const content = document.getElementById("content");
        content.style.opacity = 0;
        setTimeout(() => { content.style.display = "none"; }, 300);
        document.getElementById("refresh-btn").disabled = true;
        document.getElementById("error-card").style.display = "none";
        startStatusCycle();
    }

    function showContent(data) {
        stopStatusCycle();
        document.getElementById("loading").style.display = "none";

        if (data.error) {
            document.getElementById("error-card").style.display = "block";
            document.getElementById("error-msg").textContent = data.error;
        } else {
            document.getElementById("subtitle").textContent =
                "Vorhersage für den nächsten Handelstag · " + data.timestamp;
            document.getElementById("val-volatility").textContent = data.volatility + "%";
            document.getElementById("val-day-range").textContent  = data.day_range + "%";

            const badge = document.getElementById("pred-badge");
            badge.textContent = data.status;
            badge.className = "prediction-badge " + data.status_class;

            document.getElementById("prob-high-val").textContent = data.prob_high + "%";
            document.getElementById("prob-low-val").textContent  = data.prob_low  + "%";

            // Animate bars after a short delay so the transition is visible
            setTimeout(() => {
                document.getElementById("bar-high").style.width = data.prob_high + "%";
                document.getElementById("bar-low").style.width  = data.prob_low  + "%";
            }, 80);
        }

        const content = document.getElementById("content");
        content.style.display = "block";
        requestAnimationFrame(() => { content.style.opacity = 1; });
        document.getElementById("refresh-btn").disabled = false;
    }

    function loadPrediction() {
        showLoading();
        fetch("/api/predict")
            .then(r => r.json())
            .then(data => showContent(data))
            .catch(err => showContent({ error: err.toString() }));
    }

    loadPrediction();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict")
def predict():
    ticker = "BTC-USD"
    try:
        model, feature_view = get_resources()

        recent_data = yf.download(ticker, period="2d", interval="1d")
        if isinstance(recent_data.columns, pd.MultiIndex):
            recent_data.columns = recent_data.columns.get_level_values(0)
        recent_data.reset_index(inplace=True)
        current_day_range = float(
            (recent_data['High'].iloc[-1] - recent_data['Low'].iloc[-1])
            / recent_data['Low'].iloc[-1]
        )

        batch_data = feature_view.get_batch_data()
        latest_volatility_30d = float(batch_data['volatility_30d'].iloc[-1])

        prediction_input = pd.DataFrame([{
            'volatility_30d': latest_volatility_30d,
            'day_range_pct': current_day_range,
        }])
        prediction = model.predict(prediction_input)
        probability = model.predict_proba(prediction_input)

        return jsonify({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'volatility': round(latest_volatility_30d * 100, 2),
            'day_range': round(current_day_range * 100, 2),
            'status': "HOCH" if prediction[0] == 1 else "NIEDRIG",
            'status_class': "high" if prediction[0] == 1 else "low",
            'prob_high': round(probability[0][1] * 100, 1),
            'prob_low': round(probability[0][0] * 100, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
