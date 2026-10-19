
-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) as total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join
SELECT c.city, p.category, SUM(o.order_amount) as total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar subquery
SELECT product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common table expression
WITH CustomerTotals AS (
    SELECT customer_id, SUM(order_amount) as total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
AverageTotal AS (
    SELECT AVG(total_spend) as avg_spend
    FROM CustomerTotals
)
SELECT ct.customer_id, ct.total_spend
FROM CustomerTotals ct, AverageTotal at
WHERE ct.total_spend > at.avg_spend;

-- Q6: Ranking window functions
WITH SpendPerCust AS (
    SELECT o.customer_id, c.city, SUM(o.order_amount) as total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT 
    customer_id, 
    city, 
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rank_val,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as dense_rank_val,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) as row_num
FROM SpendPerCust;

-- Q7: Running total
WITH DailyRev AS (
    SELECT SUBSTR(order_ts, 1, 10) as day, SUM(order_amount) as revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    day, 
    revenue, 
    SUM(revenue) OVER (ORDER BY day) as cumulative_revenue
FROM DailyRev;
