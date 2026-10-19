import pandas as pd
import numpy as np
import sqlite3
import os

# Create folders if they don't exist
os.makedirs('data/clean', exist_ok=True)

# 1. Load Raw Data
print("--- AUDITING RAW DATA ---")
customers = pd.read_csv('data/raw/customers.csv')
products = pd.read_csv('data/raw/products.csv')
orders = pd.read_csv('data/raw/orders.csv')
tickets = pd.read_csv('data/raw/tickets.csv', keep_default_na=False)

# Audit Checkpoints
print(f"Raw orders rows: {len(orders)}")
print(f"Exact duplicate orders: {orders.duplicated().sum()}")
print(f"Pending rows: {(orders['status'] == 'pending').sum()}")
print(f"Distinct order_id: {orders['order_id'].nunique()}")
print(f"Unparseable order_amount: {orders['order_amount'].apply(lambda x: pd.to_numeric(x, errors='coerce')).isna().sum()}")
print(f"Negative delivery_minutes: {(pd.to_numeric(orders['delivery_minutes'], errors='coerce') < 0).sum()}")
print(f"Blank delivery_minutes: {orders['delivery_minutes'].isna().sum()}")

# 2. Clean Customers
customers['city'] = customers['city'].str.strip().str.title()
customers['gender'] = customers['gender'].str.lower().str.strip()
customers['gender'] = customers['gender'].map({
    'male': 'male', 'm': 'male', 
    'female': 'female', 'f': 'female'
}).fillna('unknown')

customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = np.nan
customers['age'] = customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# 3. Clean Orders
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
orders = orders.drop_duplicates()
orders = orders.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')

orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')

# Merge with products to calculate missing amounts and get category for delivery median
orders = orders.merge(products[['product_id', 'price_inr', 'category']], on='product_id', how='left')

# Impute order_amount
calc_amt = round(orders['price_inr'] * orders['quantity'] * (1 - orders['discount_pct'] / 100), 2)
orders['order_amount'] = orders['order_amount'].fillna(calc_amt)

# Impute delivery_minutes
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
completed = orders['status'] == 'completed'
medians = orders[completed].groupby('category')['delivery_minutes'].transform('median')
orders.loc[completed & orders['delivery_minutes'].isna(), 'delivery_minutes'] = medians

# Drop the helper columns used for calculation to match original schema
orders = orders.drop(columns=['price_inr', 'category'])

# 4. Store Data
customers.to_csv('data/clean/customers.csv', index=False)
products.to_csv('data/clean/products.csv', index=False)
orders.to_csv('data/clean/orders.csv', index=False)

conn = sqlite3.connect('data/instamart.db')
# Store order_ts as string format YYYY-MM-DD HH:MM:SS
orders_sql = orders.copy()
orders_sql['order_ts'] = orders_sql['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
customers_sql = customers.copy()
customers_sql['signup_date'] = customers_sql['signup_date'].dt.strftime('%Y-%m-%d')

customers_sql.to_sql('customers', conn, if_exists='replace', index=False)
products.to_sql('products', conn, if_exists='replace', index=False)
orders_sql.to_sql('orders', conn, if_exists='replace', index=False)
conn.close()

# 5. Validation Check
completed_orders = orders[orders['status'] == 'completed']
assert len(orders) == 12309
assert len(completed_orders) == 11110
assert (orders['status'] == 'pending').sum() == 0
assert orders['order_amount'].isna().sum() == 0
assert completed_orders['delivery_minutes'].isna().sum() == 0
assert round(completed_orders['order_amount'].sum(), 2) == 2998668.80
print("\nALL CHECKPOINTS PASSED")

# 6. Pandas EDA
print("\n--- PANDAS EDA CROSS-CHECK ---")
eda_orders = orders[orders['status'] == 'completed'].merge(customers, on='customer_id').merge(products, on='product_id')
pivot = pd.pivot_table(eda_orders, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)
print("Pivot Table Grand Total:", pivot.loc['All', 'All'])

city_stats = eda_orders.groupby('city')['order_amount'].agg(['sum', 'mean', 'count'])
print("\nCity Stats:\n", city_stats.loc[['Bengaluru', 'Mumbai']])

amt = eda_orders['order_amount']
q1 = amt.quantile(0.25)
q3 = amt.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
print(f"\nOrder Amount Mean: {amt.mean():.2f}, Median: {amt.median():.2f}, Skew: {amt.skew():.2f}")
print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, Upper Fence: {upper_fence:.2f}")
print(f"Outliers above fence: {(amt > upper_fence).sum()}")

corr = eda_orders[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr()
print("\nCorrelation Matrix:\n", corr)