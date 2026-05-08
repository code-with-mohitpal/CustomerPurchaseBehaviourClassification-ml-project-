"""
Flask API  –  Customer ML Backend
Endpoints:
  POST /api/predict        → classification + regression + cluster
  GET  /api/metrics        → model metrics
  GET  /api/sales-trend    → monthly sales data
  GET  /api/segments       → segment info
  GET  /health             → health-check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib, json, os, traceback

# ── Paths ──────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "..", "ml", "models")

app = Flask(__name__)
CORS(app)

# ── Load artefacts ─────────────────────────────────────────────────────
def load_models():
    try:
        clf      = joblib.load(os.path.join(MODELS, "classifier.pkl"))
        reg      = joblib.load(os.path.join(MODELS, "regressor.pkl"))
        km       = joblib.load(os.path.join(MODELS, "kmeans.pkl"))
        pca      = joblib.load(os.path.join(MODELS, "pca.pkl"))
        f_scaler = joblib.load(os.path.join(MODELS, "feature_scaler.pkl"))
        c_scaler = joblib.load(os.path.join(MODELS, "cluster_scaler.pkl"))
        le       = joblib.load(os.path.join(MODELS, "label_encoder.pkl"))
        features = joblib.load(os.path.join(MODELS, "features.pkl"))
        with open(os.path.join(MODELS, "metrics.json")) as f:
            metrics = json.load(f)
        return clf, reg, km, pca, f_scaler, c_scaler, le, features, metrics
    except Exception as e:
        print(f"⚠  Could not load models: {e}")
        return (None,)*9

clf, reg, km, pca, f_scaler, c_scaler, le, features, metrics_store = load_models()

SEGMENT_LABELS = [
    "Budget Shoppers",
    "Premium Buyers",
    "Occasional Browsers",
    "Loyal High-Spenders",
]
SEGMENT_COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12"]

CLUSTER_FEATS = ["Quantity","UnitPrice","NumProducts","SessionLength",
                 "PagesViewed","EngagementScore","TotalSales"]

# ── Helpers ────────────────────────────────────────────────────────────
def validate_input(data):
    required = ["Quantity","UnitPrice","NumProducts","DayOfWeek","Month",
                "Hour","IsReturning","SessionLength","PagesViewed","Country"]
    missing  = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    bounds = {
        "Quantity"     : (1, 1000),
        "UnitPrice"    : (0.01, 10000),
        "NumProducts"  : (1, 200),
        "DayOfWeek"    : (0, 6),
        "Month"        : (1, 12),
        "Hour"         : (0, 23),
        "IsReturning"  : (0, 1),
        "SessionLength": (0, 300),
        "PagesViewed"  : (1, 200),
    }
    for field, (lo, hi) in bounds.items():
        val = float(data[field])
        if not (lo <= val <= hi):
            raise ValueError(f"'{field}' must be between {lo} and {hi}, got {val}")

def build_feature_row(data):
    known_countries = list(le.classes_)
    country = data.get("Country","UK")
    if country not in known_countries:
        country = known_countries[0]
    country_enc = int(le.transform([country])[0])

    qty   = float(data["Quantity"])
    price = float(data["UnitPrice"])
    pages = float(data["PagesViewed"])
    sess  = float(data["SessionLength"])
    month = int(data["Month"])
    dow   = int(data["DayOfWeek"])

    row = {
        "Quantity"       : qty,
        "UnitPrice"      : price,
        "NumProducts"    : float(data["NumProducts"]),
        "DayOfWeek"      : dow,
        "Month"          : month,
        "Hour"           : float(data["Hour"]),
        "IsReturning"    : float(data["IsReturning"]),
        "SessionLength"  : sess,
        "PagesViewed"    : pages,
        "Country_enc"    : country_enc,
        "PricePerPage"   : price / (pages + 1),
        "EngagementScore": sess * pages / 60,
        "IsWeekend"      : 1 if dow >= 5 else 0,
        "HighSeason"     : 1 if month in [11, 12] else 0,
    }
    return pd.DataFrame([row])[features]

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": clf is not None})

@app.route("/api/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Models not loaded. Run ml_pipeline.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    try:
        validate_input(data)
        row = build_feature_row(data)

        # Scale for clf / reg
        X_scaled = f_scaler.transform(row)

        # 1. Classification
        purchase_prob = float(clf.predict_proba(X_scaled)[0][1])
        will_purchase = int(clf.predict(X_scaled)[0])

        # 2. Regression
        predicted_sales = float(max(0, reg.predict(X_scaled)[0]))

        # 3. Cluster
        qty   = float(data["Quantity"])
        price = float(data["UnitPrice"])
        pages = float(data["PagesViewed"])
        sess  = float(data["SessionLength"])
        eng   = sess * pages / 60
        total = qty * price

        cluster_row = np.array([[qty, price,
                                  float(data["NumProducts"]),
                                  sess, pages, eng, total]])
        cluster_scaled  = c_scaler.transform(cluster_row)
        cluster_id      = int(km.predict(cluster_scaled)[0])
        cluster_name    = SEGMENT_LABELS[cluster_id]
        cluster_color   = SEGMENT_COLORS[cluster_id]

        # PCA coords for visualisation
        pca_coords = pca.transform(cluster_scaled)[0].tolist()

        return jsonify({
            "will_purchase"    : will_purchase,
            "purchase_probability": round(purchase_prob * 100, 2),
            "predicted_sales"  : round(predicted_sales, 2),
            "cluster_id"       : cluster_id,
            "cluster_name"     : cluster_name,
            "cluster_color"    : cluster_color,
            "pca_coords"       : pca_coords,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception:
        return jsonify({"error": "Internal server error", "detail": traceback.format_exc()}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if not metrics_store:
        return jsonify({"error": "Metrics not available"}), 503
    return jsonify({
        "classification": metrics_store.get("classification"),
        "regression"    : metrics_store.get("regression"),
    })

@app.route("/api/sales-trend", methods=["GET"])
def sales_trend():
    if not metrics_store:
        return jsonify({"error": "Metrics not available"}), 503
    trend_raw = metrics_store.get("sales_trend", {})
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    data   = [{"month": months[int(k)-1], "sales": v}
              for k, v in sorted(trend_raw.items(), key=lambda x: int(x[0]))]
    return jsonify(data)

@app.route("/api/segments", methods=["GET"])
def get_segments():
    if not metrics_store:
        return jsonify({"error": "Metrics not available"}), 503
    segs = metrics_store.get("segments", {})
    result = []
    for sid, info in segs.items():
        result.append({
            "id"   : int(sid),
            "name" : SEGMENT_LABELS[int(sid)],
            "count": info["count"],
            "color": SEGMENT_COLORS[int(sid)],
        })
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
