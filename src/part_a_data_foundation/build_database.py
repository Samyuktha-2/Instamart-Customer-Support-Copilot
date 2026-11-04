
import sqlite3
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

CLEAN_PATH = Path("data/clean")
DB_PATH = Path("data/instamart.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD CLEANED DATASETS
# --------------------------------------------------

def load_cleaned_data():

    print("\n========== LOADING CLEANED DATA ==========")

    customers = pd.read_csv(
        CLEAN_PATH / "customers.csv"
    )

    products = pd.read_csv(
        CLEAN_PATH / "products.csv"
    )

    orders = pd.read_csv(
        CLEAN_PATH / "orders.csv"
    )

    print("Customers loaded:", len(customers))
    print("Products loaded:", len(products))
    print("Orders loaded:", len(orders))

    return customers, products, orders


# --------------------------------------------------
# PREPARE DATA TYPES
# --------------------------------------------------

def prepare_data(customers, products, orders):

    print("\n========== PREPARING DATA TYPES ==========")

    # Convert customer signup_date to date text
    customers["signup_date"] = (
        pd.to_datetime(
            customers["signup_date"],
            errors="raise"
        )
        .dt.strftime("%Y-%m-%d")
    )

    # Convert order_ts to required timestamp text format
    orders["order_ts"] = (
        pd.to_datetime(
            orders["order_ts"],
            errors="raise"
        )
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Ensure numeric columns have appropriate types
    customers["age"] = pd.to_numeric(
        customers["age"],
        errors="raise"
    )

    products["price_inr"] = pd.to_numeric(
        products["price_inr"],
        errors="raise"
    )

    orders["quantity"] = pd.to_numeric(
        orders["quantity"],
        errors="raise"
    )

    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"],
        errors="raise"
    )

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"],
        errors="raise"
    )

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"],
        errors="raise"
    )

    print("Customer signup_date:", customers["signup_date"].dtype)
    print("Order timestamp:", orders["order_ts"].dtype)

    return customers, products, orders


# --------------------------------------------------
# CREATE SQLITE DATABASE
# --------------------------------------------------

def create_database(customers, products, orders):

    print("\n========== CREATING SQLITE DATABASE ==========")

    # Connect to SQLite database
    connection = sqlite3.connect(DB_PATH)

    try:

        # Enable foreign key enforcement
        connection.execute("PRAGMA foreign_keys = ON")

        # Store cleaned DataFrames as SQLite tables
        customers.to_sql(
            "customers",
            connection,
            if_exists="replace",
            index=False
        )

        products.to_sql(
            "products",
            connection,
            if_exists="replace",
            index=False
        )

        orders.to_sql(
            "orders",
            connection,
            if_exists="replace",
            index=False
        )

        connection.commit()

        print("Customers table created successfully.")
        print("Products table created successfully.")
        print("Orders table created successfully.")

    finally:
        connection.close()


# --------------------------------------------------
# VERIFY DATABASE
# --------------------------------------------------

def verify_database():

    print("\n========== DATABASE VERIFICATION ==========")

    connection = sqlite3.connect(DB_PATH)

    try:

        # Retrieve table names
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """,
            connection
        )

        print("\nTables created:")
        print(tables)

        expected_tables = {
            "customers",
            "products",
            "orders"
        }

        actual_tables = set(tables["name"])

        assert actual_tables == expected_tables, (
            f"Unexpected tables: {actual_tables}"
        )

        # Verify row counts
        expected_counts = {
            "customers": 1200,
            "products": 40,
            "orders": 12309
        }

        for table_name, expected_count in expected_counts.items():

            result = pd.read_sql_query(
                f"SELECT COUNT(*) AS total FROM {table_name}",
                connection
            )

            actual_count = int(result.iloc[0]["total"])

            print(
                f"{table_name}: {actual_count} rows"
            )

            assert actual_count == expected_count, (
                f"{table_name}: expected {expected_count}, "
                f"got {actual_count}"
            )

        # Verify order_ts SQLite column type
        schema = pd.read_sql_query(
            "PRAGMA table_info(orders)",
            connection
        )

        timestamp_column = schema.loc[
            schema["name"] == "order_ts",
            "type"
        ]

        assert not timestamp_column.empty, (
            "order_ts column does not exist"
        )

        timestamp_type = timestamp_column.iloc[0]

        print("\norder_ts SQLite type:", timestamp_type)

        assert timestamp_type.upper() == "TEXT", (
            "order_ts must be stored as TEXT"
        )

        # Verify timestamp format
        timestamp_check = pd.read_sql_query(
            "SELECT order_ts FROM orders",
            connection
        )

        valid_timestamps = pd.to_datetime(
            timestamp_check["order_ts"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

        assert valid_timestamps.notna().all(), (
            "Some order_ts values do not match "
            "YYYY-MM-DD HH:MM:SS"
        )

        print("Timestamp format verified.")

        # Display sample timestamps
        print("\nSample order timestamps:")
        print(timestamp_check.head())

        # SQLite integrity check
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        print("\nDatabase integrity:", integrity)

        assert integrity == "ok", (
            f"Database integrity check failed: {integrity}"
        )

        print("\nDATABASE CREATED AND VERIFIED SUCCESSFULLY")

    finally:
        connection.close()


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    customers, products, orders = load_cleaned_data()

    customers, products, orders = prepare_data(
        customers,
        products,
        orders
    )

    create_database(
        customers,
        products,
        orders
    )

    verify_database()

    print("\nDatabase location:", DB_PATH)


if __name__ == "__main__":
    main()
