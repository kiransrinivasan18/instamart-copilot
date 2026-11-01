"""Validation script for Part A: asserts the reference checkpoints. Run after part_a.py."""
import sqlite3

import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

conn = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
db_total = conn.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'").fetchone()[0]
db_rows = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
conn.close()

assert len(orders) == 12309, "orders rows"
assert orders["order_id"].is_unique, "one row per order_id"
assert len(completed) == 11110, "completed orders"
assert len(cancelled) == 1199, "cancelled orders"
assert (orders["status"] == "pending").sum() == 0, "no pending rows"
assert orders["order_amount"].isna().sum() == 0, "no missing order_amount"
assert completed["delivery_minutes"].isna().sum() == 0, "completed delivery_minutes complete"
assert cancelled["delivery_minutes"].isna().sum() == 1199, "cancelled delivery_minutes stay missing"
assert round(completed["order_amount"].sum(), 2) == 2998668.80, "completed revenue"
assert customers["city"].nunique() == 6, "six cities"
assert customers["city"].value_counts().to_dict() == {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts"
assert customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts"
assert customers["age"].isna().sum() == 0, "no missing age"
assert customers["age"].between(18, 60).all(), "ages within 18-60"
assert {"customers", "products", "orders"} <= tables, "db tables"
assert db_rows == 12309 and db_total == 2998668.80, "db content matches csv"

print("ALL CHECKPOINTS PASSED")
