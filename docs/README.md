# ⚡ CustomerIQ – Customer Purchase Behaviour ML Project

> **Classification · Sales Regression · Gradient Boosting · Bayesian Optimisation · K-Means Clustering · React Dashboard**

---

## 📁 Folder Structure

```
customer-ml-project/
├── ml/
│   ├── ml_pipeline.py          # Complete ML pipeline (train + eval + save)
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # Saved .pkl model files (auto-created)
│   │   ├── classifier.pkl
│   │   ├── regressor.pkl
│   │   ├── kmeans.pkl
│   │   ├── pca.pkl
│   │   ├── feature_scaler.pkl
│   │   ├── cluster_scaler.pkl
│   │   ├── label_encoder.pkl
│   │   ├── features.pkl
│   │   └── metrics.json
│   ├── plots/                  # Auto-generated visualisations
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance_clf.png
│   │   ├── regression_scatter.png
│   │   ├── residuals.png
│   │   ├── elbow_curve.png
│   │   ├── clusters_pca.png
│   │   └── cluster_profiles.png
│   └── data/
│       └── retail_dataset.csv  # Dataset (auto-generated)
│
├── backend/
│   └── app.py                  # Flask REST API
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── App.css
│       └── pages/
│           ├── Dashboard.jsx   # Metrics + charts
│           ├── Predictor.jsx   # Real-time prediction UI
│           └── Segments.jsx    # Customer segment explorer
│
├── docs/
│   ├── README.md               # This file
│   ├── SETUP.md                # Step-by-step setup
│   ├── viva_questions.md       # 30 viva Q&A
│   └── ppt_content.md          # PPT slide content
```

---

## 🚀 Quick Start (3 terminals)

### Terminal 1 – Train ML Models
```bash
cd customer-ml-project/ml
pip install -r requirements.txt
python ml_pipeline.py
```
Wait ~3-5 min. Models + plots saved automatically.

### Terminal 2 – Start Flask API
```bash
cd customer-ml-project/backend
python app.py
# Running on http://localhost:5000
```

### Terminal 3 – Start React Dashboard
```bash
cd customer-ml-project/frontend
npm install
npm run dev
# Running on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## 🧠 ML Models Explained

| Model | Algorithm | Optimisation | Task |
|---|---|---|---|
| Classifier | XGBoost | Bayesian (BayesSearchCV) | Will customer purchase? |
| Regressor  | XGBoost | Bayesian (BayesSearchCV) | Predict total sales (₹) |
| Segmenter  | K-Means  | Elbow method (k=4)      | Customer group |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET  | `/health`           | Server health check |
| POST | `/api/predict`      | Get prediction for one customer |
| GET  | `/api/metrics`      | Model accuracy / R² metrics |
| GET  | `/api/sales-trend`  | Monthly average sales |
| GET  | `/api/segments`     | Segment counts & info |

### Sample POST `/api/predict`
```json
{
  "Quantity": 5,
  "UnitPrice": 25.99,
  "NumProducts": 3,
  "DayOfWeek": 2,
  "Month": 11,
  "Hour": 14,
  "IsReturning": 1,
  "SessionLength": 18,
  "PagesViewed": 7,
  "Country": "UK"
}
```

### Sample Response
```json
{
  "will_purchase": 1,
  "purchase_probability": 87.34,
  "predicted_sales": 129.95,
  "cluster_id": 3,
  "cluster_name": "Loyal High-Spenders",
  "cluster_color": "#f39c12"
}
```

---

## 📊 Dataset

- **Source**: Inspired by [UCI Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Auto-generated**: Realistic synthetic dataset (5000 rows) with same features
- **Features**: Quantity, UnitPrice, NumProducts, DayOfWeek, Month, Hour, IsReturning, SessionLength, PagesViewed, Country
- **Targets**: WillPurchase (classification) · TotalSales (regression)

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | Python, Pandas, NumPy, Scikit-learn, XGBoost, Scikit-optimize |
| Visualisation | Matplotlib, Seaborn |
| Backend | Flask, Flask-CORS |
| Frontend | React.js (Vite), Recharts |
| Optimisation | Bayesian Search (BayesSearchCV) |
| Clustering | K-Means + PCA |
