# 🎓 Viva Questions & Answers
## CustomerIQ – ML Project

---

### SECTION 1: Machine Learning Fundamentals

**Q1. What is the difference between Classification and Regression?**
> **A:** Classification predicts a *discrete label* (e.g., will purchase: Yes/No), while Regression predicts a *continuous numeric value* (e.g., total sales = ₹129.95). In this project, XGBoost handles both tasks — one model outputs a class, the other outputs a sales amount.

---

**Q2. Why did you choose XGBoost over Random Forest or a simple Decision Tree?**
> **A:** XGBoost (Extreme Gradient Boosting) is faster, handles missing values natively, supports regularisation (L1/L2) to prevent overfitting, and consistently wins on tabular data benchmarks. It also provides feature importance out of the box and integrates easily with hyperparameter search tools.

---

**Q3. What is Gradient Boosting and how does it work?**
> **A:** Gradient Boosting builds an ensemble of *weak learners* (shallow decision trees) sequentially. Each new tree is trained to correct the *residual errors* of the previous ensemble by computing the gradient of the loss function. The final prediction is a weighted sum of all trees. Unlike Bagging (Random Forest), Boosting is sequential, not parallel.

---

**Q4. What is overfitting, and how did you prevent it in this project?**
> **A:** Overfitting occurs when a model memorises training data and performs poorly on unseen data. We prevented it by:
> - Using **cross-validation** (CV=3) during Bayesian tuning
> - XGBoost's built-in **regularisation** (`reg_alpha`, `reg_lambda`)
> - Tuning `subsample` and `colsample_bytree` (feature/sample randomness)
> - Holding out a **20% test set** never used during training

---

**Q5. Explain Precision, Recall, and F1-Score.**
> **A:**
> - **Precision** = TP / (TP + FP) — Of all predicted "Will Purchase", how many actually did?
> - **Recall** = TP / (TP + FN) — Of all who actually purchased, how many did we detect?
> - **F1-Score** = Harmonic mean of Precision and Recall — balances both, useful when classes are imbalanced.
> In retail, Recall matters more (we don't want to miss potential buyers).

---

### SECTION 2: Hyperparameter Optimisation

**Q6. What is Bayesian Optimisation and why is it better than Grid Search?**
> **A:** Grid Search tries every combination of hyperparameters (exponential cost). Random Search samples randomly. **Bayesian Optimisation** builds a *probabilistic surrogate model* (usually Gaussian Process) of the objective function and uses it to intelligently choose the *next most promising* hyperparameter set, balancing *exploration* and *exploitation*. It finds better results in far fewer iterations (we used 25 iterations vs. thousands for grid search).

---

**Q7. What hyperparameters did you tune and what do they control?**
> **A:**
> - `n_estimators` — number of trees (more = slower but potentially better)
> - `max_depth` — maximum depth of each tree (controls model complexity)
> - `learning_rate` — shrinks contribution of each tree (lower = more robust, needs more trees)
> - `subsample` — fraction of training rows used per tree (reduces variance)
> - `colsample_bytree` — fraction of features used per tree (like Random Forest's feature sampling)

---

**Q8. What is cross-validation and why is it used?**
> **A:** Cross-validation (k-fold) splits data into k folds, trains on k-1 folds and validates on the remaining fold, rotating through all folds. The metric is averaged across folds. It gives a more *reliable estimate of generalisation performance* than a single train/test split, which can be lucky or unlucky depending on how data was split.

---

### SECTION 3: Clustering

**Q9. How does K-Means clustering work?**
> **A:** K-Means:
> 1. Randomly initialises k centroids
> 2. Assigns each data point to the nearest centroid (Euclidean distance)
> 3. Recomputes centroids as mean of assigned points
> 4. Repeats steps 2-3 until centroids stop moving (convergence)
> It minimises *within-cluster sum of squares* (WCSS / inertia).

---

**Q10. How did you choose k=4 for clustering?**
> **A:** We used the **Elbow Method** — plot WCSS (inertia) vs. k values (2–8). The "elbow" is where the rate of decrease in inertia sharply slows. At k=4, adding more clusters gives diminishing returns. Business intuition also validates 4 segments: Budget, Premium, Occasional, and Loyal customers.

---

**Q11. Why did you use PCA before visualising clusters?**
> **A:** Our clustering uses 7 features (high-dimensional). We cannot directly plot 7D data. PCA (Principal Component Analysis) reduces dimensionality to 2 principal components that capture the maximum variance, allowing us to visualise cluster separation on a 2D scatter plot.

---

**Q12. What is the difference between Supervised and Unsupervised Learning?**
> **A:** 
> - **Supervised**: Training data has labels (our classifier uses `WillPurchase`, regressor uses `TotalSales`)
> - **Unsupervised**: No labels — the algorithm finds hidden patterns. K-Means clustering is unsupervised; it discovers customer groups without us defining them.

---

### SECTION 4: Data Preprocessing

**Q13. How did you handle missing values?**
> **A:** We used **median imputation** for numerical features (Quantity, UnitPrice, SessionLength). Median is preferred over mean because it is robust to outliers — a few very high purchase values won't skew the imputed value.

---

**Q14. What is feature engineering and what new features did you create?**
> **A:** Feature engineering creates new informative variables from existing ones. We created:
> - `PricePerPage` = UnitPrice / (PagesViewed + 1) — browsing efficiency signal
> - `EngagementScore` = SessionLength × PagesViewed / 60 — overall engagement
> - `IsWeekend` = 1 if DayOfWeek ≥ 5 — captures weekend shopping behaviour
> - `HighSeason` = 1 if Month ∈ {11, 12} — captures holiday shopping season

---

**Q15. Why did you scale features before modelling?**
> **A:** XGBoost is tree-based and does NOT require scaling (trees split on thresholds, not distances). However, K-Means uses Euclidean distance, so features on larger scales (e.g., TotalSales=5000 vs IsWeekend=0/1) would dominate. We used `RobustScaler` for clustering (robust to outliers) and `StandardScaler` for the neural features.

---

### SECTION 5: Full Stack & APIs

**Q16. Why Flask for the backend?**
> **A:** Flask is lightweight, Python-native (same language as ML code), and ideal for serving ML models via REST APIs. It requires minimal boilerplate compared to Django. `Flask-CORS` enables cross-origin requests from our React frontend running on a different port.

---

**Q17. What is REST API and how does it work in this project?**
> **A:** REST (Representational State Transfer) uses HTTP methods (GET, POST) to transfer data. Our `/api/predict` endpoint accepts a `POST` request with JSON customer data, loads the trained model, runs inference, and returns JSON predictions. React fetches this data using the browser's `fetch()` API.

---

**Q18. How does the React frontend communicate with the Flask backend?**
> **A:** React uses JavaScript's `fetch()` API to send HTTP requests to Flask endpoints (e.g., `POST http://localhost:5000/api/predict`). Flask-CORS allows this cross-origin communication. React stores the JSON response in state (`useState`) and renders it dynamically in the UI.

---

### SECTION 6: Model Evaluation

**Q19. What metrics did you use to evaluate the classifier?**
> **A:**
> - **Accuracy** — overall correct predictions / total
> - **Precision** — avoid false positives (wrongly targeting non-buyers with campaigns)
> - **Recall** — avoid missing actual buyers (lost revenue opportunity)
> - **F1-Score** — harmonic mean of precision & recall (best single metric for imbalanced classes)
> - **Confusion Matrix** — visual breakdown of TP, TN, FP, FN

---

**Q20. What metrics did you use for the regression model?**
> **A:**
> - **R² (R-squared)** — proportion of variance explained (1.0 = perfect, 0 = baseline mean)
> - **RMSE** (Root Mean Squared Error) — penalises large errors more; in same units as target
> - **MAE** (Mean Absolute Error) — average absolute prediction error; easier to interpret
> - **Actual vs Predicted scatter** — visual check for systematic bias
> - **Residual histogram** — should be centred at 0 with normal distribution

---

**Q21. What is R² score and what does it mean?**
> **A:** R² measures how much of the variance in the target variable is explained by the model. R²=0.85 means the model explains 85% of the variation in sales. R²=0 means the model is no better than predicting the mean. R²=1 is perfect. Negative R² means the model is worse than predicting the mean.

---

### SECTION 7: Business & Advanced

**Q22. What business value does this project provide?**
> **A:**
> - **Classification**: Identifies high-probability buyers → targeted marketing campaigns
> - **Regression**: Forecasts expected revenue → inventory & budget planning
> - **Clustering**: Segments customers → personalised offers (discounts for Budget Shoppers, VIP for Loyal High-Spenders)
> - **Dashboard**: Real-time predictions help sales teams prioritise leads

---

**Q23. What is the difference between XGBoost and LightGBM?**
> **A:** Both are Gradient Boosting frameworks. LightGBM grows trees **leaf-wise** (faster, better on large datasets) while XGBoost grows **level-wise** (more stable). LightGBM is typically 10x faster on large datasets but may overfit on small ones. For ~5000 rows, XGBoost and LightGBM perform similarly; we used XGBoost for wider familiarity.

---

**Q24. Can you explain the bias-variance tradeoff?**
> **A:** 
> - **High Bias** (underfitting): Model is too simple, misses patterns (e.g., linear model for non-linear data)
> - **High Variance** (overfitting): Model is too complex, memorises noise (e.g., deep tree on small data)
> - **Tradeoff**: Increasing model complexity reduces bias but increases variance. Regularisation, cross-validation, and hyperparameter tuning help find the sweet spot.

---

**Q25. What is the purpose of the Confusion Matrix?**
> **A:** A confusion matrix is a 2×2 table showing:
> - **True Positive (TP)**: Predicted purchase → actually purchased ✓
> - **True Negative (TN)**: Predicted no purchase → actually didn't ✓
> - **False Positive (FP)**: Predicted purchase → actually didn't ✗ (Type I error)
> - **False Negative (FN)**: Predicted no purchase → actually did ✗ (Type II error)
> It lets us see exactly where the model fails.

---

**Q26. How would you improve this project for production?**
> **A:**
> - Use real transactional database (PostgreSQL / MongoDB)
> - Add model retraining pipeline (monthly refresh with new data)
> - Implement A/B testing to measure business impact of predictions
> - Add authentication to the API (JWT tokens)
> - Deploy to cloud (AWS / GCP / Azure) with Docker containers
> - Add drift detection (monitor when input distribution changes)
> - Use MLflow for experiment tracking

---

**Q27. What is PCA and why did you use it?**
> **A:** Principal Component Analysis is a dimensionality reduction technique. It finds orthogonal axes (principal components) that maximise variance in the data. We used it to reduce 7 clustering features to 2 dimensions purely for **visualisation** — the actual clustering was done in full 7D space. PCA preserves the most important structure of the data.

---

**Q28. Explain the `skopt` library and BayesSearchCV.**
> **A:** `scikit-optimize` (`skopt`) provides Bayesian Optimisation tools. `BayesSearchCV` is a drop-in replacement for `GridSearchCV` / `RandomizedSearchCV`. It wraps a scikit-learn estimator and uses a **Gaussian Process** surrogate model to intelligently navigate the hyperparameter space, requiring far fewer evaluations (we used 25 iterations) to find near-optimal parameters.

---

**Q29. What is `RobustScaler` and why use it over `StandardScaler`?**
> **A:** `StandardScaler` scales features using mean and standard deviation — sensitive to outliers. `RobustScaler` uses **median and interquartile range (IQR)** — robust to outliers. For clustering, retail data often has extreme purchase values, so RobustScaler prevents outliers from distorting the cluster assignments.

---

**Q30. How do you know your model is ready for deployment?**
> **A:**
> - Accuracy / R² meets business requirements (e.g., >80% accuracy)
> - Cross-validation scores are consistent (low variance between folds)
> - No significant gap between train and test performance (no overfitting)
> - Residuals are randomly distributed (no systematic error)
> - Model passes fairness checks (predictions not biased by protected attributes)
> - Latency is acceptable (< 100ms per prediction for real-time use)

---

*Prepared for college submission and placement interviews.*
