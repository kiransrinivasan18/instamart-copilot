import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------------
# TASK 1: RFM Table
# ---------------------------------------------------------
print("--- 1. RFM CALCULATION ---")
orders = pd.read_csv('data/clean/orders.csv')
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
completed = orders[orders['status'] == 'completed']

snapshot = pd.to_datetime('2025-10-01 00:00:00')
rfm = completed.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (snapshot - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

print(f"Total Customers: {len(rfm)}")
print(f"Recency - Median: {rfm['recency'].median()}, Min: {rfm['recency'].min()}, Max: {rfm['recency'].max()}")
print(f"Frequency - Mean: {rfm['frequency'].mean():.2f}, Max: {rfm['frequency'].max()}")
print(f"Monetary - Mean: {rfm['monetary'].mean():.2f}, Max: {rfm['monetary'].max()}")

# ---------------------------------------------------------
# TASK 2: K-Means Clustering
# ---------------------------------------------------------
print("\n--- 2. CLUSTERING ---")
X_rfm = rfm[['recency', 'frequency', 'monetary']]
scaler_rfm = StandardScaler()
X_scaled = scaler_rfm.fit_transform(X_rfm)

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    preds = km.fit_predict(X_scaled)
    print(f"K={k}: Inertia = {km.inertia_:.1f}, Silhouette = {silhouette_score(X_scaled, preds):.3f}")

# Selecting K=4 based on the elbow / silhouette
km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm['segment_id'] = km4.fit_predict(X_scaled)

# ---------------------------------------------------------
# TASK 3: Profile & Project
# ---------------------------------------------------------
print("\n--- 3. PROFILING & PCA ---")
pca = PCA(n_components=2)
pca.fit(X_scaled)
print(f"PCA Variance Ratios: {pca.explained_variance_ratio_}, Total: {sum(pca.explained_variance_ratio_):.3f}")

means = rfm.groupby('segment_id')[['recency', 'frequency', 'monetary']].mean()
print("\nSegment Means:\n", means)
premium_idx = means['monetary'].idxmax()
rfm['is_premium'] = rfm['segment_id'] == premium_idx

def assign_name(row):
    if row['is_premium']: return "Premium"
    if row['frequency'] > 15: return "Frequent"
    if row['recency'] > 30: return "At Risk"
    return "Occasional"

rfm['segment_name'] = rfm.apply(assign_name, axis=1)
rfm.to_csv('outputs/customer_segments.csv', index=False)

# ---------------------------------------------------------
# TASK 4: Lapse Definition & Feature Engineering
# ---------------------------------------------------------
print("\n--- 4. LAPSE DATA PREP ---")
cutoff = pd.to_datetime('2025-09-01 00:00:00')
before_cutoff = orders[orders['order_ts'] < cutoff].copy()
sept_orders = orders[(orders['order_ts'] >= cutoff) & (orders['order_ts'] < pd.to_datetime('2025-10-01 00:00:00')) & (orders['status'] == 'completed')]

cohort = before_cutoff[before_cutoff['status'] == 'completed']['customer_id'].unique()
sept_buyers = sept_orders['customer_id'].unique()

products = pd.read_csv('data/clean/products.csv')
before_cutoff = before_cutoff.merge(products[['product_id', 'category']], on='product_id', how='left')

customers = pd.read_csv('data/clean/customers.csv')
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

features = []
for cid in cohort:
    cust_orders = before_cutoff[before_cutoff['customer_id'] == cid]
    comp = cust_orders[cust_orders['status'] == 'completed']
    canc = cust_orders[cust_orders['status'] == 'cancelled']

    lapsed = 1 if cid not in sept_buyers else 0
    recency = (cutoff - comp['order_ts'].max()).days
    frequency = comp['order_id'].nunique()
    monetary = comp['order_amount'].sum()
    avg_disc = comp['discount_pct'].mean() if not comp.empty else 0
    cancel_rate = len(canc) / (len(comp) + len(canc)) if (len(comp) + len(canc)) > 0 else 0
    avg_deliv = comp['delivery_minutes'].mean() if not comp['delivery_minutes'].isna().all() else 0
    
    fav_cat = comp['category'].mode()
    fav_cat = fav_cat[0] if not fav_cat.empty else 'Unknown'

    cust_info = customers[customers['customer_id'] == cid].iloc[0]
    tenure = (cutoff - cust_info['signup_date']).days

    features.append({
        'customer_id': cid,
        'recency': recency, 'frequency': frequency, 'monetary': monetary,
        'avg_disc': avg_disc, 'cancel_rate': cancel_rate, 'avg_deliv': avg_deliv,
        'fav_cat': fav_cat, 'city': cust_info['city'], 'age': cust_info['age'],
        'gender': cust_info['gender'], 'tenure': tenure, 'lapsed': lapsed
    })

df_model = pd.DataFrame(features)
print(f"Cohort size: {len(df_model)}")
print(f"Lapsed count: {df_model['lapsed'].sum()} ({df_model['lapsed'].mean():.2%})")

# ---------------------------------------------------------
# TASK 5: Preprocessing Pipeline (No Leakage)
# ---------------------------------------------------------
print("\n--- 5. PREPROCESSING ---")
X = df_model.drop(columns=['customer_id', 'lapsed'])
y = df_model['lapsed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

categorical = ['fav_cat', 'city', 'gender']
numerical = ['recency', 'frequency', 'monetary', 'avg_disc', 'cancel_rate', 'avg_deliv', 'age', 'tenure']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
])

# Fit on training data ONLY
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

learned_freq_mean = preprocessor.named_transformers_['num'].mean_[numerical.index('frequency')]
print(f"Learned Mean Frequency (Train only): {learned_freq_mean:.2f}")
print(f"Full Cohort Mean Frequency: {df_model['frequency'].mean():.2f}")

# ---------------------------------------------------------
# TASK 6: Train, Tune, Evaluate
# ---------------------------------------------------------
print("\n--- 6. TUNING MODELS ---")
print(f"Baseline Test F1 (predicting all 0): {f1_score(y_test, np.zeros_like(y_test), zero_division=0):.3f}")

lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                       {'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1')
lr_grid.fit(X_train_proc, y_train)
print(f"LogReg - Best C: {lr_grid.best_params_}, CV F1: {lr_grid.best_score_:.3f}")

rf_grid = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42),
                       {'max_depth': [3, 5, 8], 'min_samples_leaf': [5, 10, 20]}, cv=5, scoring='f1')
rf_grid.fit(X_train_proc, y_train)
print(f"RandomForest - Best Params: {rf_grid.best_params_}, CV F1: {rf_grid.best_score_:.3f}")

y_pred_rf = rf_grid.predict(X_test_proc)
print("\nRandom Forest Test Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ---------------------------------------------------------
# TASK 7: Class Balance Experiment
# ---------------------------------------------------------
print("\n--- 7. CLASS BALANCE EXPERIMENT ---")
train_df = pd.DataFrame(X_train_proc)
train_df['y'] = y_train.values

majority = train_df[train_df['y'] == 0]
minority = train_df[train_df['y'] == 1]
minority_up = minority.sample(n=len(majority), replace=True, random_state=42)

balanced_train = pd.concat([majority, minority_up])
X_train_bal = balanced_train.drop(columns=['y']).values
y_train_bal = balanced_train['y'].values

lr_bal = LogisticRegression(C=lr_grid.best_params_['C'], max_iter=1000, random_state=42)
lr_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = lr_bal.predict(X_test_proc)

print("\nBalanced LR Test Classification Report:\n", classification_report(y_test, y_pred_bal, zero_division=0))

# ---------------------------------------------------------
# TASK 8: Interpret & Persist
# ---------------------------------------------------------
print("\n--- 8. PERSIST MODEL & RESULTS ---")
best_rf = rf_grid.best_estimator_
cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical)
all_feat_names = numerical + list(cat_names)

importances = best_rf.feature_importances_
top_features = sorted(zip(all_feat_names, importances), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 RF Features: {top_features}")

# Save full pipeline to prevent leakage in production
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_rf)
])
joblib.dump(pipeline, 'outputs/lapse_model.joblib')

loaded_pipe = joblib.load('outputs/lapse_model.joblib')

# Build retention list
test_cids = df_model.iloc[X_test.index]['customer_id'].values
test_probs = loaded_pipe.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    'customer_id': test_cids,
    'lapse_probability': test_probs,
    'actually_lapsed': y_test.values
})

results = results.merge(rfm[['customer_id', 'is_premium']], on='customer_id', how='left')
premium_retention = results[results['is_premium'] == True].sort_values('lapse_probability', ascending=False)

retention_list = premium_retention.head(10)[['customer_id', 'lapse_probability', 'actually_lapsed']]
retention_list.to_csv('outputs/retention_list.csv', index=False)
print(f"Retention list written. True lapsed in top 10: {retention_list['actually_lapsed'].sum()}")
print("\nPART B COMPLETE!")