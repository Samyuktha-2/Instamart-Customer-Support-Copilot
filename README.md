# Instamart-Customer-Support-Copilot

The objective of this project is to build a reliable customer intelligence system that converts raw operational data into actionable insights, improves customer support efficiency, and enables consistent, policy-compliant refund decisions.

# Setup and Execution

## Prerequisites

- Python 3.11 or later
- SQLite 3.25 or later
- Git
- A terminal such as PowerShell or the VS Code terminal

## 1. Clone the repository and navigate to the project root

```powershell
git clone <your-github-repository-url>
cd instamart-customer-copilot
```

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

## 4. Generate the synthetic dataset

```powershell
python generate_data.py
```

# Part A — Data Foundation and SQL Analysis

Part A establishes the data foundation for the Instamart Customer Intelligence & Support Copilot. It covers raw data auditing, data cleaning, validation, SQLite database creation, SQL-based analysis, and exploratory data analysis (EDA) using pandas.The synthetic dataset contains customer, product, order, and support ticket records. The raw data is inspected for missing values, duplicate records, inconsistent categories, invalid numeric values, and data quality issues before being cleaned and stored. The cleaned data is then used to perform seven SQL analyses and cross-check the results using pandas. Validation checkpoints ensure that the cleaned datasets and database match the expected reference results.

## Execute the Part A pipeline

Once the scripts are saved, execute the full Part A pipeline from the root using:

```powershell
python generate_data.py
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/audit_raw.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/clean_data.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/build_database.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/validate_data.py }
if ($LASTEXITCODE -eq 0) { python src/part_a_data_foundation/run_analysis.py }
```

# Part B — Customer Intelligence and Lapse Prediction

## Overview
Part B uses customer purchase history to group customers and predict who may stop purchasing.

## 1. Customer Segmentation
RFM analysis was performed on **1,175 customers**. K-means clustering grouped them into four segments:

- **Occasional Customers:** Purchase less frequently.
- **Premium Customers:** Spend the most.
- **Loyal Regulars:** Purchase frequently.
- **At-Risk Customers:** Have not purchased recently.

Results are saved in `outputs/customer_segments.csv`.

## 2. Lapse Prediction
A customer was considered lapsed if they made no completed purchase during September 2025. The dataset contained 1,156 customers, including 443 lapsed customers.

Logistic Regression and Random Forest were tested. Oversampling was used to improve the model's ability to identify lapsed customers.

## 3. Selected Model and Conclusion
**Balanced Logistic Regression** was selected because it achieved the best F1 score of **0.6972** and recall of **85.39%**.

This model identifies more potentially lapsed customers, although it may also flag some customers who are not actually lapsed. It can help prioritise customer retention campaigns.

- Final model: `outputs/lapse_model.joblib`
- Retention list: `outputs/retention_list.csv`

## Limitations
The data is synthetic, so real-world performance may differ. The model should be tested on real, future customer data before deployment.