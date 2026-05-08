"""
============================================================
 Customer Purchase Behaviour Classification,
 Sales Prediction (Regression),
 Gradient Boosting + Bayesian Tuning,
 K-Means Customer Segmentation
============================================================
Dataset : UCI Online Retail II  (auto-downloaded via opendatasets / fallback synthetic)
"""

import warnings, os, json, pickle, time
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, mean_squared_error,
                                      r2_score, mean_absolute_error)
from sklearn.cluster          import KMeans
from sklearn.decomposition    import PCA
from xgboost                  import XGBClassifier, XGBRegressor
from skopt                    import BayesSearchCV
from skopt.space              import Real, Integer
import joblib

# ─────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models");  os.makedirs(MODELS, exist_ok=True)
PLOTS  = os.path.join(BASE, "plots");   os.makedirs(PLOTS,  exist_ok=True)
DATA   = os.path.join(BASE, "data");    os.makedirs(DATA,   exist_ok=True)

# ─────────────────────────────────────────────
# 1.  DATA GENERATION  (realistic synthetic retail)
# ─────────────────────────────────────────────
def generate_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    countries = ["UK","Germany","France","Netherlands","Australia","Japan","Sweden","Norway"]
    df = pd.DataFrame({
        "CustomerID"    : rng.integers(10000, 20000, n),
        "Country"       : rng.choice(countries, n, p=[.55,.10,.10,.05,.05,.05,.05,.05]),
        "Quantity"      : rng.integers(1, 50, n).astype(float),
        "UnitPrice"     : np.round(rng.uniform(0.5, 150, n), 2),
        "NumProducts"   : rng.integers(1, 20, n),
        "DayOfWeek"     : rng.integers(0, 7, n),
        "Month"         : rng.integers(1, 13, n),
        "Hour"          : rng.integers(8, 21, n),
        "IsReturning"   : rng.integers(0, 2, n),
        "SessionLength" : np.round(rng.uniform(1, 60, n), 1),
        "PagesViewed"   : rng.integers(1, 30, n),
    })
    # Introduce ~5 % missing
    for col in ["Quantity","UnitPrice","SessionLength"]:
        idx = rng.choice(n, int(n*0.05), replace=False)
        df.loc[idx, col] = np.nan

    df["TotalSales"] = (df["Quantity"].fillna(df["Quantity"].median()) *
                        df["UnitPrice"].fillna(df["UnitPrice"].median()))
    # Classification target
    threshold = df["TotalSales"].median()
    df["WillPurchase"] = (df["TotalSales"] > threshold).astype(int)
    return df

# ─────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    # Missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical
    le = LabelEncoder()
    df["Country_enc"] = le.fit_transform(df["Country"])
    joblib.dump(le, os.path.join(MODELS, "label_encoder.pkl"))

    # Feature engineering
    df["PricePerPage"]    = df["UnitPrice"] / (df["PagesViewed"] + 1)
    df["EngagementScore"] = df["SessionLength"] * df["PagesViewed"] / 60
    df["IsWeekend"]       = (df["DayOfWeek"] >= 5).astype(int)
    df["HighSeason"]      = df["Month"].isin([11, 12]).astype(int)

    features = ["Quantity","UnitPrice","NumProducts","DayOfWeek","Month","Hour",
                "IsReturning","SessionLength","PagesViewed","Country_enc",
                "PricePerPage","EngagementScore","IsWeekend","HighSeason"]
    return df, features

# ─────────────────────────────────────────────
# 3.  CLASSIFICATION  (XGBoost + Bayesian Opt)
# ─────────────────────────────────────────────
def train_classifier(X_train, X_test, y_train, y_test):
    print("\n[1/4] 🔷 Classification  –  XGBoost + Bayesian Tuning")

    search_space = {
        "n_estimators"     : Integer(50, 300),
        "max_depth"        : Integer(3, 10),
        "learning_rate"    : Real(0.01, 0.3, prior="log-uniform"),
        "subsample"        : Real(0.5, 1.0),
        "colsample_bytree" : Real(0.5, 1.0),
    }
    base = XGBClassifier(eval_metric="logloss", use_label_encoder=False,
                         random_state=42, n_jobs=-1)
    bayes = BayesSearchCV(base, search_space, n_iter=25, cv=3,
                          scoring="accuracy", n_jobs=-1, random_state=42, verbose=0)
    bayes.fit(X_train, y_train)

    best = bayes.best_estimator_
    y_pred = best.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"   Best params : {bayes.best_params_}")
    print(f"   Accuracy    : {acc:.4f}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Purchase","Purchase"],
                yticklabels=["No Purchase","Purchase"], ax=ax)
    ax.set_title("Classification – Confusion Matrix", fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance
    fi = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    fi.head(10).plot.barh(ax=ax, color="#4f86c6")
    ax.set_title("Top-10 Feature Importances (Classifier)", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"feature_importance_clf.png"), dpi=150)
    plt.close()

    joblib.dump(best, os.path.join(MODELS,"classifier.pkl"))
    metrics = {"accuracy": round(acc,4),
               "precision": round(report["1"]["precision"],4),
               "recall"   : round(report["1"]["recall"],4),
               "f1"       : round(report["1"]["f1-score"],4),
               "best_params": {k:float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v
                               for k,v in bayes.best_params_.items()}}
    return best, metrics

# ─────────────────────────────────────────────
# 4.  REGRESSION  (XGBoost + Bayesian Opt)
# ─────────────────────────────────────────────
def train_regressor(X_train, X_test, y_train, y_test):
    print("\n[2/4] 🔶 Regression  –  XGBoost + Bayesian Tuning")

    search_space = {
        "n_estimators"  : Integer(50, 300),
        "max_depth"     : Integer(3, 8),
        "learning_rate" : Real(0.01, 0.3, prior="log-uniform"),
        "subsample"     : Real(0.5, 1.0),
    }
    base  = XGBRegressor(random_state=42, n_jobs=-1)
    bayes = BayesSearchCV(base, search_space, n_iter=25, cv=3,
                          scoring="r2", n_jobs=-1, random_state=42, verbose=0)
    bayes.fit(X_train, y_train)

    best   = bayes.best_estimator_
    y_pred = best.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print(f"   Best params : {bayes.best_params_}")
    print(f"   R²          : {r2:.4f}  |  RMSE : {rmse:.2f}  |  MAE : {mae:.2f}")

    # Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(y_test[:300], y_pred[:300], alpha=0.4, color="#e07b39", edgecolors="none", s=20)
    lim = max(y_test.max(), y_pred.max())
    ax.plot([0, lim],[0, lim], "k--", lw=1)
    ax.set_xlabel("Actual Sales"); ax.set_ylabel("Predicted Sales")
    ax.set_title(f"Regression – Actual vs Predicted  (R²={r2:.3f})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"regression_scatter.png"), dpi=150)
    plt.close()

    # Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(residuals, bins=50, color="#6abf69", edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.set_xlabel("Residual"); ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"residuals.png"), dpi=150)
    plt.close()

    joblib.dump(best, os.path.join(MODELS,"regressor.pkl"))
    metrics = {"rmse": round(float(rmse),2), "mae": round(float(mae),2),
               "r2"  : round(float(r2),4),
               "best_params": {k:float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v
                               for k,v in bayes.best_params_.items()}}
    return best, metrics

# ─────────────────────────────────────────────
# 5.  K-MEANS CLUSTERING
# ─────────────────────────────────────────────
def train_clustering(df, features):
    print("\n[3/4] 🔵 K-Means Clustering")

    cluster_feats = ["Quantity","UnitPrice","NumProducts","SessionLength",
                     "PagesViewed","EngagementScore","TotalSales"]
    X_clust = df[cluster_feats].fillna(df[cluster_feats].median())
    scaler  = RobustScaler()
    X_scaled = scaler.fit_transform(X_clust)
    joblib.dump(scaler, os.path.join(MODELS,"cluster_scaler.pkl"))

    # Elbow
    inertias = []
    K_range  = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(list(K_range), inertias, "o-", color="#9c27b0", lw=2, ms=7)
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve for K-Means", fontweight="bold")
    ax.axvline(4, color="red", ls="--", lw=1.5, label="Chosen k=4")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"elbow_curve.png"), dpi=150)
    plt.close()

    # Final model k=4
    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels   = km_final.fit_predict(X_scaled)
    df["Cluster"] = labels
    joblib.dump(km_final, os.path.join(MODELS,"kmeans.pkl"))

    # PCA 2-D scatter
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(MODELS,"pca.pkl"))

    PALETTE = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
    LABELS  = ["Budget Shoppers","Premium Buyers","Occasional Browsers","Loyal High-Spenders"]

    fig, ax = plt.subplots(figsize=(8,6))
    for c in range(4):
        mask = labels == c
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=PALETTE[c], label=LABELS[c], alpha=0.55, s=18, edgecolors="none")
    ax.set_title("Customer Segments – PCA Projection", fontweight="bold", fontsize=13)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"clusters_pca.png"), dpi=150)
    plt.close()

    # Segment profile bar charts
    profile = df.groupby("Cluster")[cluster_feats].mean().round(2)
    profile.index = LABELS
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for ax, col in zip(axes, ["TotalSales","SessionLength","NumProducts"]):
        profile[col].plot.bar(ax=ax, color=PALETTE, edgecolor="white")
        ax.set_title(col, fontweight="bold"); ax.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=8)
    plt.suptitle("Cluster Profiles", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,"cluster_profiles.png"), dpi=150)
    plt.close()

    seg_counts = pd.Series(labels).value_counts().sort_index()
    seg_info = {int(i): {"name": LABELS[i], "count": int(seg_counts[i])} for i in range(4)}
    print(f"   Segment counts : {seg_counts.to_dict()}")
    return km_final, pca, scaler, cluster_feats, LABELS, seg_info

# ─────────────────────────────────────────────
# 6.  SALES TREND (for dashboard)
# ─────────────────────────────────────────────
def generate_sales_trend(df):
    trend = df.groupby("Month")["TotalSales"].mean().round(2)
    return {int(k): float(v) for k,v in trend.items()}

# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("  ML Pipeline – Customer Analytics")
    print("=" * 60)

    # Data
    print("\n⏳  Generating dataset …")
    df = generate_dataset(5000)
    df.to_csv(os.path.join(DATA,"retail_dataset.csv"), index=False)
    print(f"   Shape : {df.shape}")

    df, features = preprocess(df)

    X = df[features]
    y_clf = df["WillPurchase"]
    y_reg = df["TotalSales"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    joblib.dump(scaler, os.path.join(MODELS,"feature_scaler.pkl"))
    joblib.dump(features, os.path.join(MODELS,"features.pkl"))

    X_tr, X_te, yc_tr, yc_te = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
    _, _, yr_tr, yr_te        = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    clf,  clf_metrics  = train_classifier(X_tr, X_te, yc_tr, yc_te)
    reg,  reg_metrics  = train_regressor (X_tr, X_te, yr_tr, yr_te)
    km, pca, cs, cf, seg_labels, seg_info = train_clustering(df, features)

    # Sales trend
    sales_trend = generate_sales_trend(df)

    # Persist all metrics
    all_metrics = {
        "classification" : clf_metrics,
        "regression"     : reg_metrics,
        "segments"       : seg_info,
        "sales_trend"    : sales_trend,
        "segment_features": cf,
        "segment_labels" : seg_labels,
    }
    with open(os.path.join(MODELS,"metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[4/4] ✅  Done in {time.time()-t0:.1f}s")
    print(f"   Models → {MODELS}")
    print(f"   Plots  → {PLOTS}")
    print("=" * 60)

if __name__ == "__main__":
    main()
