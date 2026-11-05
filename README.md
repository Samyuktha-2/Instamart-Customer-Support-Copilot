# Instamart Customer Support Copilot

## Overview

A customer intelligence and support automation system that analyses operational data, segments customers, predicts customer lapse, classifies support tickets, and processes policy-compliant refund decisions.

**Technologies:** Python, SQLite, Pandas, Scikit-learn, TF-IDF and offline `mock_llm.py`.

## Reproduce

### Prerequisites
- Python 3.11+
- SQLite 3.25+
- Git

### Setup

```powershell
git clone <your-github-repository-url>
cd Instamart-Customer-Support-Copilot

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python generate_data.py
```

### Part A execution

```powershell
python src/part_a_data_foundation/audit_raw.py
python src/part_a_data_foundation/clean_data.py
python src/part_a_data_foundation/build_database.py
python src/part_a_data_foundation/validate_data.py
python src/part_a_data_foundation/run_analysis.py
```

Parts B–D use the generated datasets and database. Run their scripts from `src/` in the required pipeline order.

All LLM interactions use the supplied offline `mock_llm.py`. No API key is required.

## Part A — Data Foundation and SQL Analysis

Audited, cleaned and validated synthetic customer, product, order and ticket data. Built a SQLite database and performed seven SQL analyses, cross-validated using Pandas.

**Results:**
- Unique orders: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Revenue: INR 2,998,668.80
- Database integrity: Passed

## Part B — Customer Intelligence and Lapse Prediction

### Customer Segmentation

Applied RFM analysis and K-means clustering to 1,175 customers.

| Segment | Customers |
|---|---:|
| Occasional Customers | 436 |
| Premium Customers | 138 |
| Loyal Regulars | 419 |
| At-Risk Customers | 182 |

### Lapse Prediction

Defined lapse as no completed purchase during September 2025.

- Lapse cohort: 1,156
- Lapsed customers: 443
- Non-lapsed customers: 713

Balanced Logistic Regression was selected after evaluating Logistic Regression and Random Forest.

**Performance:**
- Accuracy: 71.55%
- Precision: 0.5891
- Recall: 85.39%
- F1 score: 0.6972

**Outputs:** `customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv`.

## Part C — Ticket Triage and Support Routing

Implemented ticket classification using offline `mock_llm.py` with PII redaction, JSON validation, retries and fallback handling.

**Data processing:**
- Tickets: 164
- Eligible model calls: 156
- Classified: 147
- Valid JSON: 135
- Keyword classifications: 12
- Invalid responses: 9
- Empty tickets: 4
- Long tickets: 4
- PII redactions: 278 (55 emails, 91 phone numbers, 132 order IDs)

### Zero-Shot vs Few-Shot

| Metric | Zero-Shot | Few-Shot |
|---|---:|---:|
| Accuracy | 85.7% | 100% |
| Classified | 147 | 147 |
| Tokens | 16,888 | 31,084 |

Few-shot improved accuracy by 14.3 percentage points, using approximately 84.1% more tokens.

A 6,000-token budget demonstration classified 53 tickets, used 6 retries and reached 6,062 tokens.

### Final Routing

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| premium_desk | 11 |
| human_review | 17 |

## Part D — Policy Assistant and Refund ReAct Agent

### Policy RAG Assistant

Built a TF-IDF and cosine similarity retrieval system over five policy documents.

- Chunk size: 40 words
- Overlap: 10 words
- Total chunks: 15
- Retrieved chunks: Top 3

| Metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Document recall | 12/12 | 12/12 |
| Fact accuracy | 12/12 | 10/12 |
| Unsupported fallback | 2/2 | 2/2 |

### Refund ReAct Agent

Implemented five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`.

Refund rules: completed orders only; delivery above 45 minutes qualifies for a full refund; refunds above INR 1,000 require human approval; requests follow the seven-day policy window.

**Scenario results:**

| Scenario | Status | Refund (INR) | Calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,457 |
| S2 | REFUND | 135.00 | 5 | 1,457 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,444 |
| S4 | DENIED | 0 | 4 | 1,053 |
| S5 | DENIED | 0 | 3 | 673 |
| S6 | NOT_FOUND | 0 | 2 | 296 |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3,672 |

All seven scenarios matched their expected outcomes.

### Reliability and Recovery

Implemented iteration, token and timeout safeguards. Verified:
- Token limit: `FALLBACK_TOKEN_LIMIT` (200-token limit; 2 calls, 326 tokens).
- Timeout: `FALLBACK_TIMEOUT` (0 calls).
- Iteration limit: `FALLBACK_MAX_ITERATIONS` (6 calls).

Checkpoint/resume was tested with S1. The resumed run completed four tool executions without repeating previous executions.

**Outputs:** `agent_runs.jsonl`, `refund_ledger.jsonl` and `state/` checkpoints.

### ReAct vs Plan-and-Execute

ReAct selects tools dynamically using intermediate observations. S1 required 5 model calls, S5 required 3 and S6 required 2. This supports conditional decisions for completed, cancelled and missing orders.

Plan-and-Execute creates a plan before tool execution. It can reduce repeated reasoning in predictable workflows but is less adaptive to unexpected observations. ReAct was selected because refund decisions depend on order status, delivery time and refund eligibility.

## Limitations and Next Steps

- Synthetic data may not represent real-world behaviour.
- The supplied mock LLM is not a production model.
- TF-IDF may miss semantically similar queries.
- Validate models using real data before deployment.
- Future improvements: semantic retrieval, production LLM integration, expanded evaluation and monitoring.