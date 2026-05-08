# 📽 PPT Slide Content
## CustomerIQ — Customer Purchase Behaviour ML Project

---

### SLIDE 1 — Title Slide
**Title:** CustomerIQ: End-to-End Customer Analytics with Machine Learning
**Subtitle:** Classification · Sales Regression · Gradient Boosting · Bayesian Optimisation · K-Means Clustering
**Visual:** Dark dashboard screenshot / gradient background with neural network motif
**Names / Roll Numbers / College / Date**

---

### SLIDE 2 — Problem Statement
**Title:** The Business Problem
**Points:**
- Retailers lose crores annually by targeting the wrong customers
- Manual customer segmentation is slow and inaccurate
- No data-driven way to forecast sales or identify high-value buyers
**Solution:** Build an intelligent ML system that predicts purchase intent, forecasts sales, and segments customers automatically
**Visual:** Funnel diagram — Visitors → Prospects → Buyers

---

### SLIDE 3 — Project Objectives
**Title:** What We Built
**4 pillars (2×2 grid):**
1. 🎯 **Purchase Classification** — Predict if a customer will buy
2. 📈 **Sales Regression** — Forecast the transaction value
3. ⚡ **Gradient Boosting** — High-performance XGBoost models
4. 👥 **Customer Segmentation** — K-Means clustering into 4 groups

---

### SLIDE 4 — Tech Stack
**Title:** Technology Stack
**Table:**
| Layer | Technology |
|---|---|
| ML Models | XGBoost, Scikit-learn |
| Optimisation | Bayesian Search (scikit-optimize) |
| Clustering | K-Means + PCA |
| Data | Pandas, NumPy |
| Backend | Flask (Python REST API) |
| Frontend | React.js + Recharts |
| Visualisation | Matplotlib, Seaborn |
**Visual:** Technology logos arranged in architecture layers

---

### SLIDE 5 — Dataset Overview
**Title:** Dataset & Features
**Key stats:**
- 5,000 customer sessions
- 10 raw features → 14 engineered features
- Inspired by UCI Online Retail II dataset
**Feature table:**
| Feature | Description |
|---|---|
| Quantity | Items in cart |
| UnitPrice | Price per item |
| Country | Customer location |
| SessionLength | Time on site (min) |
| PagesViewed | Pages browsed |
| IsReturning | New vs returning |
| HighSeason | Nov/Dec holiday flag |
**Visual:** Sample data table / feature distribution plots

---

### SLIDE 6 — Data Preprocessing
**Title:** Data Cleaning & Feature Engineering
**Steps:**
1. **Missing Values** → Median imputation (5% of records)
2. **Encoding** → LabelEncoder for Country
3. **Scaling** → StandardScaler (models), RobustScaler (clustering)
4. **New Features Created:**
   - `PricePerPage` = UnitPrice / PagesViewed
   - `EngagementScore` = SessionLength × PagesViewed / 60
   - `IsWeekend` = DayOfWeek ≥ 5
   - `HighSeason` = Month ∈ {11, 12}
**Visual:** Before/after data cleaning diagram

---

### SLIDE 7 — Classification Model
**Title:** Purchase Classification with XGBoost
**Content:**
- **Task:** Predict `WillPurchase` (0 or 1)
- **Algorithm:** XGBoost Classifier
- **Optimisation:** Bayesian Search (25 iterations, 3-fold CV)
- **Split:** 80% train / 20% test
**Results Box:**
- ✅ Accuracy: ~85%
- ✅ F1-Score: ~0.84
- ✅ Precision: ~86%
- ✅ Recall: ~83%
**Visual:** Confusion matrix heatmap

---

### SLIDE 8 — Regression Model
**Title:** Sales Prediction with XGBoost Regressor
**Content:**
- **Task:** Predict `TotalSales` (continuous ₹ value)
- **Algorithm:** XGBoost Regressor
- **Optimisation:** Bayesian Search (25 iterations)
**Results Box:**
- ✅ R² Score: ~0.88
- ✅ RMSE: ~₹45
- ✅ MAE: ~₹28
**Visual:** Actual vs Predicted scatter plot + Residual histogram

---

### SLIDE 9 — Bayesian Optimisation
**Title:** Hyperparameter Tuning with Bayesian Optimisation
**Why Bayesian over Grid Search?**
- Grid Search: tries ALL combinations → O(nᵏ) evaluations
- Bayesian: builds surrogate model → finds best params in ~25 evaluations
**Tuned Parameters:**
- `n_estimators`, `max_depth`, `learning_rate`
- `subsample`, `colsample_bytree`
**Visual:** Comparison chart — Grid Search vs Bayesian iterations vs accuracy

---

### SLIDE 10 — Customer Segmentation
**Title:** K-Means Customer Segmentation
**Method:**
1. Select 7 behavioural features
2. Apply RobustScaler
3. Elbow method → k = 4
4. Visualise with PCA (2D projection)
**4 Segments:**
- 🔴 Budget Shoppers — price sensitive, promo driven
- 🔵 Premium Buyers — brand loyal, high basket value
- 🟢 Occasional Browsers — low frequency, low conversion
- 🟡 Loyal High-Spenders — repeat buyers, VIP candidates
**Visual:** Cluster PCA scatter plot (coloured)

---

### SLIDE 11 — Visualisations Gallery
**Title:** Model Insights
**Grid of 4 plots:**
1. Confusion Matrix
2. Feature Importance (Top 10)
3. Actual vs Predicted Sales
4. Customer Cluster PCA Scatter
**Caption:** All generated automatically by ml_pipeline.py

---

### SLIDE 12 — Full Stack Architecture
**Title:** System Architecture
**Flow Diagram:**
```
User Input (React UI)
       ↓
React Frontend (Vite)
       ↓ HTTP POST
Flask REST API (Python)
       ↓
ML Models (XGBoost + K-Means)
       ↓
JSON Response
       ↓
React Dashboard (Charts + Results)
```
**Visual:** Architecture diagram with arrows and tech icons

---

### SLIDE 13 — Dashboard Demo
**Title:** Live Dashboard Screenshot
**3 Pages:**
1. **📊 Dashboard** — Metric tiles, sales trend chart, classifier metrics bar chart
2. **🔮 Predictor** — Input form → instant purchase prediction + sales forecast + segment
3. **👥 Segments** — Pie chart, radar chart, segment profiles + recommended actions
**Visual:** Large dashboard screenshots (dark theme)

---

### SLIDE 14 — Results Summary
**Title:** Project Results
**Comparison Table:**
| Metric | Value |
|---|---|
| Classifier Accuracy | ~85% |
| F1 Score | ~0.84 |
| Regression R² | ~0.88 |
| Regression RMSE | ~₹45 |
| Segments Found | 4 |
| Bayesian Iterations | 25 (vs 1000+ Grid) |
| API Response Time | < 50ms |

---

### SLIDE 15 — Future Scope
**Title:** What's Next?
**Points:**
1. 🚀 Deploy to AWS / GCP with Docker
2. 🔄 Automated monthly retraining pipeline
3. 📱 Mobile app (React Native)
4. 🤖 Add LightGBM / CatBoost comparison
5. 📊 MLflow experiment tracking
6. 🔐 JWT authentication for API
7. 📉 Model drift detection & alerting
8. 💬 NLP on customer reviews for sentiment-based segmentation

---

### SLIDE 16 — Conclusion
**Title:** Key Takeaways
**Points:**
- Built a complete end-to-end ML + Full Stack project from scratch
- Applied 3 ML techniques: Classification, Regression, Clustering
- Used state-of-the-art Gradient Boosting (XGBoost) + Bayesian Optimisation
- Delivered business-ready insights via React dashboard
- Project architecture mirrors real-world ML product deployment

**Closing line:** *"Data is the new oil — and this project is the refinery."*

---

### SLIDE 17 — Thank You / Q&A
**Title:** Thank You!
**Content:**
- GitHub link / Code repository
- Dataset source: UCI Online Retail II
- Team names and acknowledgements
**Visual:** Large "⚡" icon + "CustomerIQ" branding
