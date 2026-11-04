
-- ============================================================
-- PART A: SQL ANALYSIS
-- Database: data/instamart.db
-- ============================================================


-- Q1. Customers whose completed orders above INR 500
--     have a combined amount greater than INR 8,000

SELECT
    customer_id,
    SUM(order_amount) AS total_amount
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_amount DESC;


-- ============================================================
-- Q2. Revenue by city and product category
-- ============================================================

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY
    c.city,
    p.category
ORDER BY
    c.city,
    p.category;


-- ============================================================
-- Q3. Products priced above the average catalogue price
-- ============================================================

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- ============================================================
-- Q4. Customers with no completed orders
-- ============================================================

SELECT
    c.customer_id,
    c.city,
    c.gender
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- ============================================================
-- Q5. Customers whose completed-order spending is above
--     the average spending per customer
-- ============================================================

WITH customer_spending AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spending AS (
    SELECT
        AVG(total_spend) AS avg_spend
    FROM customer_spending
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend
FROM customer_spending AS cs
CROSS JOIN average_spending AS av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;


-- ============================================================
-- Q6. Rank customers by completed-order spending within
--     each city using RANK, DENSE_RANK, and ROW_NUMBER
-- ============================================================

WITH customer_spending AS (
    SELECT
        c.city,
        o.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM orders AS o
    JOIN customers AS c
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY
        c.city,
        o.customer_id
)
SELECT
    city,
    customer_id,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC, customer_id
    ) AS spend_row_number
FROM customer_spending
ORDER BY
    city,
    spend_rank,
    customer_id;


-- ============================================================
-- Q7. Daily completed-order revenue and cumulative revenue
-- ============================================================

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(daily_revenue, 2) AS daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
