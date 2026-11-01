"""Part A: audit, clean, store and cross-check the Instamart data."""
import os
import sqlite3

import numpy as np
import pandas as pd

os.makedirs("data/clean", exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# ---------------------------------------------------------------- 2. AUDIT
print("=== AUDIT: shape, dtypes, missing values per column ===")
for name, df in [("customers", customers), ("products", products), ("orders", orders), ("tickets", tickets)]:
    print(f"\n[{name}] shape = {df.shape}")
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(rep.to_string())

age_num = pd.to_numeric(customers["age"], errors="coerce")
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()),
    "negative delivery_minutes": int((pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(age_num.isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank raw gender spellings": int(customers["gender"].nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES ===")
for k, v in audit.items():
    print(f"| {k} | {v:,} |")

# ---------------------------------------------------------------- 3. CLEAN CUSTOMERS
print("\n=== CLEANING: CUSTOMERS ===")
city_raw = customers["city"].copy()
customers["city"] = customers["city"].str.strip().str.title()
print(f"city rows changed by strip+title-case: {int((city_raw != customers['city']).sum())}; distinct cities now: {customers['city'].nunique()}")

gender_raw = customers["gender"].copy()
gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
customers["gender"] = customers["gender"].str.strip().str.lower().map(gender_map).fillna("unknown")
print(f"gender rows changed by mapping: {int((gender_raw != customers['gender']).sum())} (of which blank -> unknown: {int(gender_raw.isna().sum())})")

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
n_blank_age = int(customers["age"].isna().sum())
bad_age = (customers["age"] < 10) | (customers["age"] > 100)
customers.loc[bad_age, "age"] = np.nan
n_missing_age = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
print(f"ages outside 10-100 set to missing: {int(bad_age.sum())}; ages filled with city median: {n_missing_age}; missing ages left: {int(customers['age'].isna().sum())}")
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---------------------------------------------------------------- 4. CLEAN ORDERS
print("\n=== CLEANING: ORDERS (in order) ===")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
n0 = len(orders)

orders = orders.drop_duplicates()
print(f"rule 2 - exact duplicate rows removed: {n0 - len(orders)} (rows left {len(orders)})")

n1 = len(orders)
orders = orders.sort_values("order_ts", kind="stable").drop_duplicates(subset=["order_id"], keep="last")
print(f"rule 3 - older records dropped (keep latest order_ts per order_id): {n1 - len(orders)} (rows left {len(orders)})")

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders = orders.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")

n_amt = int(orders["order_amount"].isna().sum())
calc_amt = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc_amt)
print(f"rule 4 - missing order_amount filled from product price: {n_amt}")

neg = orders["delivery_minutes"] < 0
orders.loc[neg, "delivery_minutes"] = np.nan
print(f"rule 5a - negative delivery_minutes set to missing: {int(neg.sum())}")
completed = orders["status"] == "completed"
med = orders[completed].groupby("category")["delivery_minutes"].transform("median")
to_fill = completed & orders["delivery_minutes"].isna()
print(f"rule 5b - completed orders with delivery_minutes imputed by category median: {int(to_fill.sum())}")
orders.loc[to_fill, "delivery_minutes"] = med
print(f"rule 5c - cancelled orders left with missing delivery_minutes: {int(((orders['status'] == 'cancelled') & orders['delivery_minutes'].isna()).sum())}")
orders = orders.drop(columns=["price_inr", "category"])

print(f"\nclean orders: {len(orders)} rows; status counts: {orders['status'].value_counts().to_dict()}")

# ---------------------------------------------------------------- 5. STORE
customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders.to_csv("data/clean/orders.csv", index=False)

orders_sql = orders.copy()
orders_sql["order_ts"] = orders_sql["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers_sql = customers.copy()
customers_sql["signup_date"] = customers_sql["signup_date"].dt.strftime("%Y-%m-%d")
conn = sqlite3.connect("data/instamart.db")
customers_sql.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_sql.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
print("stored: data/clean/*.csv and data/instamart.db (customers, products, orders)")

# ---------------------------------------------------------------- 6. PANDAS CROSS-CHECK + EDA
print("\n=== PANDAS CROSS-CHECK ===")
eda = orders[orders["status"] == "completed"].merge(customers, on="customer_id").merge(products, on="product_id")
pivot = pd.pivot_table(eda, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print(f"pivot grand total: {pivot.loc['All', 'All']:.2f}")
cell = pivot.drop(index="All", columns="All").stack()
print(f"largest city/category cell: {cell.idxmax()} = {cell.max():.2f}")

city_stats = eda.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")).round(2)
print(city_stats.to_string())

amt = eda["order_amount"]
q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"mean {amt.mean():.2f} | median {amt.median():.2f} | skew {amt.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f} | lower fence {lo:.2f} | upper fence {hi:.2f}")
print(f"outliers below lower fence: {int((amt < lo).sum())} | above upper fence: {int((amt > hi).sum())}")

corr = eda[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
top = pairs.abs().idxmax()
print(f"largest |correlation|: {top[0]} vs {top[1]} = {pairs[top]:.3f}")
