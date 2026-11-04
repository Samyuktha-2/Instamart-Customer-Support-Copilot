
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

DB_PATH = Path("data/instamart.db")
CLEAN_PATH = Path("data/clean")
SQL_PATH = Path("sql/queries.sql")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def check(condition, message):
    """Raise an error when an expected result does not match."""
    assert condition, f"FAILED: {message}"
    print(f"PASS: {message}")


def load_cleaned_data():
    """Load the cleaned CSV files for pandas analysis."""
    customers = pd.read_csv(CLEAN_PATH / "customers.csv")
    products = pd.read_csv(CLEAN_PATH / "products.csv")
    orders = pd.read_csv(CLEAN_PATH / "orders.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="raise"
    )

    return customers, products, orders


def load_sql_queries():
    """Read and separate the seven SQL statements."""
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    queries = [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
        and any(
            line.strip() and not line.strip().startswith("--")
            for line in statement.splitlines()
        )
    ]

    check(len(queries) == 7, "Exactly seven SQL queries found")

    return queries


# --------------------------------------------------
# EXECUTE SQL QUERIES
# --------------------------------------------------

def run_sql_analysis(connection, queries):
    print("\n========== SQL ANALYSIS ==========")

    results = {}

    for index, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)
        results[index] = result

        print(f"\nQ{index} result: {len(result)} rows")
        print(result.head(5).to_string(index=False))

    return results


# --------------------------------------------------
# VALIDATE SQL RESULTS
# --------------------------------------------------

def validate_sql_results(results):
    print("\n========== SQL RESULT VALIDATION ==========")

    q1 = results[1]
    q2 = results[2]
    q3 = results[3]
    q4 = results[4]
    q5 = results[5]
    q6 = results[6]
    q7 = results[7]

    check(len(q1) == 30, "Q1 returns 30 customers")

    check(len(q2) == 30, "Q2 returns 30 city-category combinations")

    q2_total = q2["total_revenue"].sum()

    check(
        np.isclose(q2_total, 2998668.80, atol=0.01),
        "Q2 total revenue equals INR 2,998,668.80"
    )

    largest_cell = q2.loc[q2["total_revenue"].idxmax()]

    check(
        largest_cell["city"] == "Bengaluru"
        and largest_cell["category"] == "Electronics"
        and np.isclose(
            largest_cell["total_revenue"], 233909.45, atol=0.01
        ),
        "Q2 largest revenue cell is Bengaluru / Electronics"
    )

    check(len(q3) == 11, "Q3 returns 11 above-average-priced products")

    check(len(q4) == 25, "Q4 returns 25 customers without completed orders")

    check(
        len(q5) == 550,
        "Q5 returns 550 above-average-spending customers"
    )

    # Independently calculate the average spending across
    # customers who have at least one completed order.
    with sqlite3.connect(DB_PATH) as connection:
        average_spend = pd.read_sql_query(
            """
            SELECT AVG(total_spend) AS avg_spend
            FROM (
                SELECT customer_id, SUM(order_amount) AS total_spend
                FROM orders
                WHERE status = 'completed'
                GROUP BY customer_id
            )
            """,
            connection
        )["avg_spend"].iloc[0]

    check(
        round(average_spend, 2) == 2552.06,
        "Q5 average customer spend equals INR 2,552.06"
    )

    # Expected top-spending customer for each city.
    expected_city_leaders = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }

    city_leaders = (
        q6.sort_values(
            ["city", "spend_rank", "customer_id"]
        )
        .groupby("city", as_index=False)
        .first()
    )

    for city, (expected_customer, expected_spend) in (
        expected_city_leaders.items()
    ):
        actual = city_leaders.loc[
            city_leaders["city"] == city
        ]

        check(not actual.empty, f"Q6 has a leader for {city}")

        row = actual.iloc[0]

        check(
            row["customer_id"] == expected_customer
            and np.isclose(
                row["total_spend"], expected_spend, atol=0.01
            ),
            f"Q6 top customer for {city} matches reference"
        )

    rank_mismatches = int(
        (q6["spend_rank"] != q6["spend_row_number"]).sum()
    )

    check(
        rank_mismatches == 8,
        "Q6 has exactly 8 rows where RANK differs from ROW_NUMBER"
    )

    check(len(q7) == 183, "Q7 returns 183 days")

    final_cumulative = q7["cumulative_revenue"].iloc[-1]

    check(
        np.isclose(final_cumulative, 2998668.80, atol=0.01),
        "Q7 final cumulative revenue equals INR 2,998,668.80"
    )

    print("\nAll SQL result checkpoints passed.")


# --------------------------------------------------
# PANDAS CROSS-CHECK: REVENUE PIVOT
# --------------------------------------------------

def pandas_revenue_crosscheck(customers, products, orders, sql_q2):
    print("\n========== PANDAS REVENUE CROSS-CHECK ==========")

    completed = orders.loc[
        orders["status"] == "completed"
    ].copy()

    merged = (
        completed
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="left",
            validate="many_to_one"
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
            validate="many_to_one"
        )
    )

    pivot = pd.pivot_table(
        merged,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All"
    )

    print("\nRevenue pivot table with margins:")
    print(pivot.round(2).to_string())

    pandas_total = float(
        merged["order_amount"].sum()
    )

    sql_total = float(
        sql_q2["total_revenue"].sum()
    )

    check(
        np.isclose(pandas_total, sql_total, atol=0.01),
        "Pandas revenue total matches SQL revenue total"
    )

    check(
        np.isclose(pivot.loc["All", "All"], 2998668.80, atol=0.01),
        "Pandas pivot grand total equals INR 2,998,668.80"
    )

    # Named aggregation: completed revenue, average order amount,
    # and completed order count for each city.
    city_summary = (
        completed
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="left",
            validate="many_to_one"
        )
        .groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            average_order_amount=("order_amount", "mean"),
            completed_order_count=("order_id", "count")
        )
        .reset_index()
        .sort_values("city")
    )

    print("\nCompleted order summary by city:")
    print(city_summary.round(2).to_string(index=False))

    expected_city_stats = {
        "Bengaluru": (523643.40, 276.47, 1894),
        "Mumbai": (479058.50, 297.00, 1613),
    }

    for city, expected in expected_city_stats.items():
        row = city_summary.loc[
            city_summary["city"] == city
        ].iloc[0]

        check(
            np.isclose(row["total_revenue"], expected[0], atol=0.01)
            and np.isclose(
                row["average_order_amount"], expected[1], atol=0.01
            )
            and int(row["completed_order_count"]) == expected[2],
            f"{city} completed-order statistics match reference"
        )


# --------------------------------------------------
# PANDAS EDA: DISTRIBUTION AND OUTLIERS
# --------------------------------------------------

def analyze_order_amounts(orders):
    print("\n========== ORDER AMOUNT EDA ==========")

    completed = orders.loc[
        orders["status"] == "completed",
        "order_amount"
    ].dropna()

    q1 = completed.quantile(0.25)
    median = completed.median()
    q3 = completed.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    mean_value = completed.mean()
    skew_value = completed.skew()

    lower_outliers = int(
        (completed < lower_fence).sum()
    )

    upper_outliers = int(
        (completed > upper_fence).sum()
    )

    print(f"Completed order count: {len(completed)}")
    print(f"Mean: INR {mean_value:.2f}")
    print(f"Median: INR {median:.2f}")
    print(f"Skewness: {skew_value:.2f}")
    print(f"Q1: INR {q1:.2f}")
    print(f"Q3: INR {q3:.2f}")
    print(f"IQR: INR {iqr:.2f}")
    print(f"Lower fence: INR {lower_fence:.2f}")
    print(f"Upper fence: INR {upper_fence:.2f}")
    print(f"Outliers below lower fence: {lower_outliers}")
    print(f"Outliers above upper fence: {upper_outliers}")

    check(
        np.isclose(mean_value, 269.91, atol=0.01),
        "Completed order amount mean matches reference"
    )

    check(
        np.isclose(median, 135.00, atol=0.01),
        "Completed order amount median matches reference"
    )

    check(
        np.isclose(skew_value, 4.61, atol=0.02),
        "Completed order amount skewness matches reference"
    )

    check(
        np.isclose(q1, 64.80, atol=0.01)
        and np.isclose(q3, 297.00, atol=0.01),
        "Completed order amount quartiles match reference"
    )

    check(
        np.isclose(lower_fence, -283.50, atol=0.01)
        and np.isclose(upper_fence, 645.30, atol=0.01),
        "Completed order amount IQR fences match reference"
    )

    check(
        lower_outliers == 0 and upper_outliers == 897,
        "Completed order amount outlier counts match reference"
    )


# --------------------------------------------------
# PANDAS EDA: CORRELATION
# --------------------------------------------------

def analyze_correlations(orders):
    print("\n========== CORRELATION ANALYSIS ==========")

    completed = orders.loc[
        orders["status"] == "completed"
    ].copy()

    numeric_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]

    correlation = completed[numeric_columns].corr()

    print("\nCorrelation matrix:")
    print(correlation.round(3).to_string())

    amount_correlations = (
        correlation["order_amount"]
        .drop("order_amount")
    )

    strongest_feature = amount_correlations.abs().idxmax()
    strongest_value = amount_correlations[strongest_feature]

    print(
        "\nLargest absolute correlation with order_amount:",
        f"{strongest_feature} ({strongest_value:.3f})"
    )

    check(
        np.isclose(
            correlation.loc["order_amount", "discount_pct"],
            -0.095,
            atol=0.015
        ),
        "Order amount vs discount correlation matches reference"
    )

    check(
        np.isclose(
            correlation.loc["order_amount", "quantity"],
            0.087,
            atol=0.015
        ),
        "Order amount vs quantity correlation matches reference"
    )


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    print("========== INSTAMART PART A ANALYSIS ==========")

    for path in [DB_PATH, SQL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                "Run this script from the project root."
            )

    customers, products, orders = load_cleaned_data()
    queries = load_sql_queries()

    with sqlite3.connect(DB_PATH) as connection:
        results = run_sql_analysis(connection, queries)

    validate_sql_results(results)

    pandas_revenue_crosscheck(
        customers,
        products,
        orders,
        results[2]
    )

    analyze_order_amounts(orders)
    analyze_correlations(orders)

    print("\n========== PART A ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
