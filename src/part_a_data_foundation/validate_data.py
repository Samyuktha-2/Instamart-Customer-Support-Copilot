
import sqlite3
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

CLEAN_PATH = Path("data/clean")
DB_PATH = Path("data/instamart.db")


# --------------------------------------------------
# VALIDATION HELPER
# --------------------------------------------------

def check(condition, message):
    """Raise an error if a validation checkpoint fails."""
    assert condition, f"VALIDATION FAILED: {message}"


# --------------------------------------------------
# LOAD CLEANED DATA
# --------------------------------------------------

def load_cleaned_data():
    print("\n========== LOADING CLEANED DATA ==========")

    check(CLEAN_PATH.exists(), f"Missing directory: {CLEAN_PATH}")
    check(DB_PATH.exists(), f"Missing database: {DB_PATH}")

    customers = pd.read_csv(CLEAN_PATH / "customers.csv")
    products = pd.read_csv(CLEAN_PATH / "products.csv")
    orders = pd.read_csv(CLEAN_PATH / "orders.csv")

    print("Customers loaded:", len(customers))
    print("Products loaded:", len(products))
    print("Orders loaded:", len(orders))

    return customers, products, orders


# --------------------------------------------------
# VALIDATE CUSTOMERS
# --------------------------------------------------

def validate_customers(customers):
    print("\n========== VALIDATING CUSTOMERS ==========")

    # Row count
    check(
        len(customers) == 1200,
        f"Expected 1200 customers, got {len(customers)}"
    )

    # Required columns
    required_columns = {
        "customer_id",
        "age",
        "gender",
        "city",
        "signup_date"
    }

    check(
        required_columns.issubset(customers.columns),
        "One or more required customer columns are missing"
    )

    # Customer IDs must be unique
    check(
        customers["customer_id"].is_unique,
        "Customer IDs are not unique"
    )

    # Expected city distribution
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184
    }

    actual_cities = customers["city"].value_counts().to_dict()

    check(
        actual_cities == expected_cities,
        f"Unexpected city distribution: {actual_cities}"
    )

    # City values must be cleaned
    check(
        customers["city"].notna().all(),
        "Customer city contains missing values"
    )

    check(
        customers["city"].eq(customers["city"].str.strip()).all(),
        "Customer city contains leading or trailing spaces"
    )

    # Expected gender distribution
    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30
    }

    actual_genders = customers["gender"].value_counts().to_dict()

    check(
        actual_genders == expected_genders,
        f"Unexpected gender distribution: {actual_genders}"
    )

    check(
        customers["gender"].isin(
            ["male", "female", "unknown"]
        ).all(),
        "Unexpected gender values found"
    )

    # Age validation
    customers["age"] = pd.to_numeric(
        customers["age"],
        errors="raise"
    )

    check(
        customers["age"].notna().all(),
        "Customer age contains missing values"
    )

    check(
        customers["age"].between(10, 100).all(),
        "Customer age outside the allowed range 10–100"
    )

    check(
        customers["age"].between(18, 60).all(),
        "Customer ages do not match the expected cleaned range 18–60"
    )

    # Signup date validation
    signup_dates = pd.to_datetime(
        customers["signup_date"],
        errors="coerce"
    )

    check(
        signup_dates.notna().all(),
        "Invalid or missing signup_date values found"
    )

    print("Customer count: 1200")
    print("City distribution:", actual_cities)
    print("Gender distribution:", actual_genders)
    print(
        "Age range:",
        int(customers["age"].min()),
        "to",
        int(customers["age"].max())
    )
    print("Customer validation passed.")


# --------------------------------------------------
# VALIDATE PRODUCTS
# --------------------------------------------------

def validate_products(products):
    print("\n========== VALIDATING PRODUCTS ==========")

    check(
        len(products) == 40,
        f"Expected 40 products, got {len(products)}"
    )

    required_columns = {
        "product_id",
        "category",
        "price_inr"
    }

    check(
        required_columns.issubset(products.columns),
        "One or more required product columns are missing"
    )

    check(
        products["product_id"].is_unique,
        "Product IDs are not unique"
    )

    products["price_inr"] = pd.to_numeric(
        products["price_inr"],
        errors="raise"
    )

    check(
        products["price_inr"].notna().all(),
        "Product prices contain missing values"
    )

    check(
        (products["price_inr"] > 0).all(),
        "Product prices must be positive"
    )

    print("Product count: 40")
    print("Product validation passed.")


# --------------------------------------------------
# VALIDATE ORDERS
# --------------------------------------------------

def validate_orders(orders):
    print("\n========== VALIDATING ORDERS ==========")

    check(
        len(orders) == 12309,
        f"Expected 12309 orders, got {len(orders)}"
    )

    required_columns = {
        "order_id",
        "customer_id",
        "product_id",
        "order_ts",
        "quantity",
        "discount_pct",
        "order_amount",
        "status",
        "delivery_minutes"
    }

    check(
        required_columns.issubset(orders.columns),
        "One or more required order columns are missing"
    )

    # One row per order ID
    check(
        orders["order_id"].is_unique,
        "Duplicate order IDs found"
    )

    # Order timestamp validation
    order_timestamps = pd.to_datetime(
        orders["order_ts"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    check(
        order_timestamps.notna().all(),
        "Invalid or missing order timestamps found"
    )

    # Order status validation
    expected_statuses = {
        "completed": 11110,
        "cancelled": 1199
    }

    actual_statuses = orders["status"].value_counts().to_dict()

    check(
        actual_statuses == expected_statuses,
        f"Unexpected order status distribution: {actual_statuses}"
    )

    check(
        not orders["status"].eq("pending").any(),
        "Pending orders remain after cleaning"
    )

    # Numeric columns
    for column in [
        "quantity",
        "discount_pct",
        "order_amount",
        "delivery_minutes"
    ]:
        orders[column] = pd.to_numeric(
            orders[column],
            errors="raise"
        )

    # Order amounts must be complete
    check(
        orders["order_amount"].notna().all(),
        "Missing order_amount values found"
    )

    # No negative delivery times
    check(
        not (orders["delivery_minutes"].dropna() < 0).any(),
        "Negative delivery_minutes values found"
    )

    completed = orders[
        orders["status"] == "completed"
    ]

    cancelled = orders[
        orders["status"] == "cancelled"
    ]

    # Completed orders must have delivery times
    check(
        completed["delivery_minutes"].notna().all(),
        "Completed orders have missing delivery times"
    )

    # Cancelled orders must retain missing delivery times
    check(
        cancelled["delivery_minutes"].isna().all(),
        "Cancelled orders should have missing delivery times"
    )

    check(
        int(cancelled["delivery_minutes"].isna().sum()) == 1199,
        "Expected 1199 cancelled orders with missing delivery times"
    )

    # Completed-order revenue
    completed_revenue = round(
        completed["order_amount"].sum(),
        2
    )

    expected_revenue = 2998668.80

    check(
        completed_revenue == expected_revenue,
        f"Expected completed revenue INR {expected_revenue:,.2f}, "
        f"got INR {completed_revenue:,.2f}"
    )

    print("Unique orders:", orders["order_id"].nunique())
    print("Order status distribution:", actual_statuses)
    print(
        "Completed orders with missing delivery times:",
        int(completed["delivery_minutes"].isna().sum())
    )
    print(
        "Cancelled orders with missing delivery times:",
        int(cancelled["delivery_minutes"].isna().sum())
    )
    print(f"Completed revenue: INR {completed_revenue:,.2f}")
    print("Order validation passed.")


# --------------------------------------------------
# VALIDATE SQLITE DATABASE
# --------------------------------------------------

def validate_database():
    print("\n========== VALIDATING SQLITE DATABASE ==========")

    connection = sqlite3.connect(DB_PATH)

    try:
        # Verify required tables
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

        actual_tables = {row[0] for row in table_rows}
        expected_tables = {"customers", "products", "orders"}

        check(
            actual_tables == expected_tables,
            f"Unexpected database tables: {actual_tables}"
        )

        # Verify row counts
        expected_counts = {
            "customers": 1200,
            "products": 40,
            "orders": 12309
        }

        for table_name, expected_count in expected_counts.items():
            actual_count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            check(
                actual_count == expected_count,
                f"{table_name}: expected {expected_count}, "
                f"got {actual_count}"
            )

        # Verify order_ts is stored as TEXT
        schema = connection.execute(
            "PRAGMA table_info(orders)"
        ).fetchall()

        timestamp_columns = [
            column for column in schema
            if column[1] == "order_ts"
        ]

        check(
            len(timestamp_columns) == 1,
            "order_ts column does not exist or is duplicated"
        )

        check(
            timestamp_columns[0][2].upper() == "TEXT",
            "order_ts must be stored as TEXT"
        )

        # Verify database integrity
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        check(
            integrity == "ok",
            f"Database integrity check failed: {integrity}"
        )

        # Cross-check completed revenue in SQLite
        database_revenue = connection.execute(
            """
            SELECT ROUND(SUM(order_amount), 2)
            FROM orders
            WHERE status = 'completed'
            """
        ).fetchone()[0]

        check(
            database_revenue is not None
            and round(database_revenue, 2) == 2998668.80,
            f"Unexpected database completed revenue: {database_revenue}"
        )

        print("Required tables verified.")
        print("Database row counts verified.")
        print("order_ts TEXT storage verified.")
        print("Database integrity:", integrity)
        print(f"Database completed revenue: INR {database_revenue:,.2f}")
        print("Database validation passed.")

    finally:
        connection.close()


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    customers, products, orders = load_cleaned_data()

    validate_customers(customers)
    validate_products(products)
    validate_orders(orders)
    validate_database()

    print("\nALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
