"""Part B: RFM segments, K-means, PCA and lapse prediction."""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------ 1. RFM
print("--- 1. RFM TABLE ---")
orders = pd.read_csv("data/clean/orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
completed = orders[orders["status"] == "completed"]

snapshot = pd.Timestamp("2025-10-01 00:00:00")
rfm = completed.groupby("customer_id").agg(
    last_ts=("order_ts", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum")).reset_index()
rfm["recency"] = (snapshot - rfm["last_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print(f"customers: {len(rfm)}")
print(f"recency median {rfm['recency'].median():.0f}, min {rfm['recency'].min()}, max {rfm['recency'].max()}")
print(f"frequency mean {rfm['frequency'].mean():.2f}, max {rfm['frequency'].max()}")
print(f"monetary mean {rfm['monetary'].mean():.2f}, max {rfm['monetary'].max():.2f}")

# ------------------------------------------------------------------ 2. K-MEANS
print("\n--- 2. K-MEANS ---")
X_scaled = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    print(f"K={k}: inertia {km.inertia_:.1f}, silhouette {silhouette_score(X_scaled, labels):.3f}")

K = 4
rfm["segment_id"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_scaled)

# ------------------------------------------------------------------ 3. PROFILE, NAME, PCA
print("\n--- 3. SEGMENT PROFILE ---")
profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                        frequency=("frequency", "mean"), monetary=("monetary", "mean"))

# one name per CLUSTER, decided from the cluster's mean profile (never row by row)
premium_id = profile["monetary"].idxmax()
rest = profile.drop(index=premium_id)
at_risk_id = rest["recency"].idxmax()
loyal_id = rest.drop(index=at_risk_id)["frequency"].idxmax()
names = {i: "Occasional Buyers" for i in profile.index}
names[premium_id], names[at_risk_id], names[loyal_id] = "Premium", "At Risk", "Loyal Regulars"
actions = {"Premium": "VIP perks and early access; proactive retention outreach",
           "Loyal Regulars": "Subscription / bundle offers to protect frequency",
           "At Risk": "Win-back discount with a time limit",
           "Occasional Buyers": "Nudges and category recommendations to raise order frequency"}
profile["segment_name"] = profile.index.map(names)
profile["action"] = profile["segment_name"].map(actions)
print(profile.round(2).to_string())
print("Note: cluster numbers are only labels and have no ordering.")

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2).fit(X_scaled)
evr = pca.explained_variance_ratio_
print(f"PCA explained variance ratio: {evr[0]:.3f}, {evr[1]:.3f}; total {evr.sum():.3f}")

# ------------------------------------------------------------------ 4. LAPSE DATA
print("\n--- 4. LAPSE PROBLEM ---")
cutoff = pd.Timestamp("2025-09-01 00:00:00")
october = pd.Timestamp("2025-10-01 00:00:00")
before = orders[orders["order_ts"] < cutoff].copy()
sept = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < october) & (orders["status"] == "completed")]

products = pd.read_csv("data/clean/products.csv")
before = before.merge(products[["product_id", "category"]], on="product_id", how="left")
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"]).set_index("customer_id")

cohort = before[before["status"] == "completed"]["customer_id"].unique()
sept_buyers = set(sept["customer_id"])
assert before["order_ts"].max() < cutoff, "leakage: an order on/after the cutoff entered the features"

rows = []
for cid, grp in before[before["customer_id"].isin(cohort)].groupby("customer_id"):
    comp = grp[grp["status"] == "completed"]
    canc = grp[grp["status"] == "cancelled"]
    fav = comp["category"].mode()
    rows.append({
        "customer_id": cid,
        "recency": (cutoff - comp["order_ts"].max()).days,
        "frequency": comp["order_id"].nunique(),
        "monetary": comp["order_amount"].sum(),
        "avg_disc": comp["discount_pct"].mean(),
        "cancel_rate": len(canc) / (len(comp) + len(canc)),
        "avg_deliv": comp["delivery_minutes"].mean(),
        "fav_cat": fav.iloc[0] if not fav.empty else "Unknown",
        "city": customers.loc[cid, "city"],
        "age": customers.loc[cid, "age"],
        "gender": customers.loc[cid, "gender"],
        "tenure": (cutoff - customers.loc[cid, "signup_date"]).days,
        "lapsed": int(cid not in sept_buyers),
    })
df = pd.DataFrame(rows)
print(f"cohort {len(df)}; lapsed {df['lapsed'].sum()} ({df['lapsed'].mean():.2%})")

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]
assert "segment_id" not in X.columns and "segment_name" not in X.columns
assert X.isna().sum().sum() == 0, "feature matrix contains missing values"
print("confirmed: no feature column contains a missing value")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(X_train)}, test {len(X_test)}")
baseline_pred = np.zeros(len(y_test), dtype=int)
print(f"baseline (always not lapsed): accuracy {accuracy_score(y_test, baseline_pred):.1%}, "
      f"lapsed-class F1 {f1_score(y_test, baseline_pred, zero_division=0):.3f}")

# ------------------------------------------------------------------ 5. PREPROCESS
print("\n--- 5. PREPROCESSING ---")
categorical = ["fav_cat", "city", "gender"]
numerical = ["recency", "frequency", "monetary", "avg_disc", "cancel_rate", "avg_deliv", "age", "tenure"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical)])
Xtr = preprocessor.fit_transform(X_train)          # fitted on training rows only
Xte = preprocessor.transform(X_test)               # only applied to test rows
assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()
learned = preprocessor.named_transformers_["num"].mean_[numerical.index("frequency")]
print(f"scaler mean of frequency (train rows only): {learned:.2f} | full-cohort mean: {df['frequency'].mean():.2f}")
print("They differ because the scaler saw only the 924 training rows, not the whole cohort.")
feat_names = numerical + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical))

# ------------------------------------------------------------------ 6. TUNE + EVALUATE
print("\n--- 6. TUNING ---")
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_grid.fit(Xtr, y_train)
rf_grid = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42),
                       {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_grid.fit(Xtr, y_train)
print(f"LogReg best {lr_grid.best_params_}, CV F1 {lr_grid.best_score_:.3f}")
print(f"RandForest best {rf_grid.best_params_}, CV F1 {rf_grid.best_score_:.3f}")


def lapsed_metrics(model_or_pred, label):
    pred = model_or_pred if isinstance(model_or_pred, np.ndarray) else model_or_pred.predict(Xte)
    cm = confusion_matrix(y_test, pred)
    p, r, f = (precision_score(y_test, pred, zero_division=0), recall_score(y_test, pred, zero_division=0),
               f1_score(y_test, pred, zero_division=0))
    print(f"{label}: precision {p:.3f}, recall {r:.3f}, F1 {f:.3f}; confusion [[TN FP],[FN TP]] = {cm.tolist()}")
    return p, r, f


print("\nTest-set evaluation (lapsed class):")
lr_m = lapsed_metrics(lr_grid.best_estimator_, "LogReg")
rf_m = lapsed_metrics(rf_grid.best_estimator_, "RandForest")

# model choice is made on cross-validated F1 (tuning data), then reported on the test set
if rf_grid.best_score_ >= lr_grid.best_score_:
    chosen_name, chosen = "RandomForest", rf_grid.best_estimator_
else:
    chosen_name, chosen = "LogisticRegression", lr_grid.best_estimator_
print(f"selected model: {chosen_name}")

# ------------------------------------------------------------------ 7. CLASS BALANCE
print("\n--- 7. CLASS-BALANCE EXPERIMENT ---")
ytr = y_train.to_numpy()
minority_rows = Xtr[ytr == 1]
n_add = int((ytr == 0).sum() - (ytr == 1).sum())
extra_idx = pd.Series(np.arange(len(minority_rows))).sample(n=n_add, replace=True, random_state=42).to_numpy()
X_bal = np.vstack([Xtr, minority_rows[extra_idx]])     # every original training row is kept
y_bal = np.concatenate([ytr, np.ones(n_add, dtype=int)])
print(f"balanced training set: class 0 = {(y_bal == 0).sum()}, class 1 = {(y_bal == 1).sum()}")
lr_bal = LogisticRegression(C=lr_grid.best_params_["C"], max_iter=1000).fit(X_bal, y_bal)
unb = lapsed_metrics(lr_grid.best_estimator_, "LogReg un-balanced")
bal = lapsed_metrics(lr_bal, "LogReg balanced  ")
moves = {"precision": bal[0] - unb[0], "recall": bal[1] - unb[1], "F1": bal[2] - unb[2]}
print("change (balanced - un-balanced): " + ", ".join(f"{k} {v:+.3f}" for k, v in moves.items()))
print(f"metric that moved most: {max(moves, key=lambda k: abs(moves[k]))}")

# ------------------------------------------------------------------ 8. INTERPRET, PERSIST, USE
print("\n--- 8. INTERPRET AND PERSIST ---")
imp = sorted(zip(feat_names, rf_grid.best_estimator_.feature_importances_), key=lambda t: t[1], reverse=True)[:5]
print("top-5 random-forest features: " + ", ".join(f"{n} ({v:.3f})" for n, v in imp))
coefs = dict(zip(feat_names, lr_grid.best_estimator_.coef_[0]))
for n in ["recency", "frequency", "cancel_rate"]:
    print(f"logistic coefficient {n}: {coefs[n]:+.3f} ({'raises' if coefs[n] > 0 else 'lowers'} lapse risk)")

pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", chosen)])
original_pred = pipeline.predict(X_test)
joblib.dump(pipeline, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(reloaded.predict(X_test), original_pred)
print("reloaded model reproduces the original test predictions exactly: True")

probs = reloaded.predict_proba(X_test)[:, 1]
res = pd.DataFrame({"customer_id": df.loc[X_test.index, "customer_id"].values,
                    "lapse_probability": probs, "actually_lapsed": y_test.values})
res = res.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")
top10 = res[res["is_premium"] == True].sort_values("lapse_probability", ascending=False).head(10)  # noqa: E712
top10[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv("outputs/retention_list.csv", index=False)
print(f"retention list: {len(top10)} premium test customers, {int(top10['actually_lapsed'].sum())} truly lapsed")
print("The retention list is illustrative and not a leakage-free evaluation (Premium flag uses the whole window).")
print("\nPART B COMPLETE")
