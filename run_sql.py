"""Runs the seven queries stored in sql/queries.sql against data/instamart.db and checks the reference results."""
import re
import sqlite3

import pandas as pd

with open("sql/queries.sql", encoding="utf-8") as f:
    text = f.read()

# one query per block; each block starts with a '-- Qn' comment line
blocks = [b.strip() for b in re.split(r"(?m)^-- Q\d+.*$", text) if b.strip()]
queries = [b.rstrip(";").strip() for b in blocks]
assert len(queries) == 7, f"expected 7 queries, found {len(queries)}"

conn = sqlite3.connect("data/instamart.db")
res = {i: pd.read_sql_query(q, conn) for i, q in enumerate(queries, 1)}


def check(label, ok):
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}")


q = res[1]
print(f"Q1: {len(q)} customers")
check("reference 30", len(q) == 30)

q = res[2]
print(f"Q2: {len(q)} rows; total revenue INR {q['total_revenue'].sum():,.2f}")
top = q.loc[q["total_revenue"].idxmax()]
print(f"    largest cell: {top['city']} / {top['category']} = INR {top['total_revenue']:,.2f}")
check("reference 30 rows, 2,998,668.80, Bengaluru/Electronics 233,909.45",
      len(q) == 30 and round(q["total_revenue"].sum(), 2) == 2998668.80
      and (top["city"], top["category"], round(top["total_revenue"], 2)) == ("Bengaluru", "Electronics", 233909.45))

q = res[3]
print(f"Q3: {len(q)} products above average price")
check("reference 11", len(q) == 11)

q = res[4]
print(f"Q4: {len(q)} customers with no completed order")
check("reference 25", len(q) == 25)

q = res[5]
avg_spend = pd.read_sql_query(
    "SELECT AVG(t) AS a FROM (SELECT SUM(order_amount) AS t FROM orders WHERE status='completed' GROUP BY customer_id)", conn)["a"][0]
print(f"Q5: {len(q)} customers above the average total spend of INR {avg_spend:,.2f}")
check("reference 550, 2,552.06", len(q) == 550 and round(avg_spend, 2) == 2552.06)

q = res[6]
tops = q[q["row_num"] == 1].sort_values("city")
print(f"Q6: {len(q)} rows; top customer per city:")
for _, r in tops.iterrows():
    print(f"    {r['city']}: {r['customer_id']} ({r['total_spend']:,.2f})")
diff = int((q["rank_val"] != q["row_num"]).sum())
print(f"    rows where RANK() != ROW_NUMBER(): {diff}")
expected = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40), "Delhi": ("C0291", 13742.60),
            "Hyderabad": ("C0743", 12730.05), "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}
got = {r["city"]: (r["customer_id"], round(r["total_spend"], 2)) for _, r in tops.iterrows()}
check("reference top customers and 8 rank differences", got == expected and diff == 8)

q = res[7]
print(f"Q7: {len(q)} daily rows; last cumulative revenue INR {q['cumulative_revenue'].iloc[-1]:,.2f}")
check("reference 183 rows, 2,998,668.80", len(q) == 183 and round(q["cumulative_revenue"].iloc[-1], 2) == 2998668.80)
conn.close()
