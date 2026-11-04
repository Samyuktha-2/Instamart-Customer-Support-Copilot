# Instamart-Customer-Support-Copilot
THe objective of this project is to build a reliable customer intelligence system that converts raw operational data into actionable insights, improves customer support efficiency, and enables consistent, policy-compliant refund decisions.

# Setup and Execution

Prerequisites
    - Python 3.11 or later
    - SQLite 3.25 or later
    - Git
    - A terminal such as PowerShell or the VS Code terminal

1. Clone the repository and navigate to the project root
    git clone <your-github-repository-url>
    cd instamart-customer-copilot

2. Create and activate a virtual environment
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

3. Install dependencies

4. Generate the synthetic dataset
    python generate_data.py

# Part A — Data Foundation and SQL Analysis

Part A establishes the data foundation for the Instamart Customer Intelligence & Support Copilot. It covers raw data auditing, data cleaning, validation, SQLite database creation, SQL-based analysis, and exploratory data analysis (EDA) using pandas.

The synthetic dataset contains customer, product, order, and support ticket records. The raw data is inspected for missing values, duplicate records, inconsistent categories, invalid numeric values, and data quality issues before being cleaned and stored.

The cleaned data is then used to perform seven SQL analyses and cross-check the results using pandas. Validation checkpoints ensure that the cleaned datasets and database match the expected reference results.

Once the scripts are saved, execute the full Part A pipeline from the root using:
python generate_data.py
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/audit_raw.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/clean_data.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/build_database.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/validate_data.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/run_analysis.py }