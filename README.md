#  DataMind — AI Data Analysis Agent

An AI-powered data analysis system that automatically explores datasets, generates insights, creates visualizations, runs SQL queries, and produces a structured report — all in one click.

---

## Overview

DataMind automates the complete data analysis workflow. Upload a dataset (CSV, Excel, or SQLite), and the system analyzes it using a multi-step AI pipeline — delivering charts, statistics, SQL insights, and a final written report.

---

## Application Screens

** Overview Dashboard**

<img width="1907" height="964" alt="image" src="https://github.com/user-attachments/assets/eed64c37-7e96-4924-bca6-b0f4e4248e5d" />
Displays dataset KPIs: rows, columns, missing values, duplicates, and column averages.

** Charts**
<img width="1910" height="975" alt="image" src="https://github.com/user-attachments/assets/34766590-d30b-43d1-9f9b-59deaa11ca5d" />
- Histogram — distribution of numeric columns
- Bar Charts — top categories by count or value
- Line Charts — trends over time or ordered data
- Scatter Plots — relationships between numeric columns
- Heatmaps — correlation matrix for all numeric columns

** AI Report**
<img width="1911" height="975" alt="image" src="https://github.com/user-attachments/assets/5862abd1-e8c1-4bb5-bc87-7d4cdabe387c" />
- Correlation insights
- Distribution analysis
- Data quality assessment
- Business recommendations

** SQL Results**
<img width="1911" height="969" alt="image" src="https://github.com/user-attachments/assets/9b2caa86-fda2-4f35-acc7-d4222becab27" />
Auto-planned queries: top products, regional performance, discount analysis, and more.

** Stats**
<img width="1911" height="926" alt="image" src="https://github.com/user-attachments/assets/118dec4a-e162-4497-8f56-ca46868dd794" />
Column types, missing value counts, and full numeric summary statistics.

** SQL Editor**
<img width="1915" height="933" alt="image" src="https://github.com/user-attachments/assets/b7372185-8764-48da-8cf6-8aa0f06f4721" />
Run custom SQL queries directly on your uploaded data.

---

## How It Works

1. Upload a dataset (CSV, Excel, or SQLite)
2. AI agent reads the schema and generates an analysis plan
3. Pipeline executes: **Planner → Stats → Charts → SQL → Reporter**
4. Results displayed across the 6-tab dashboard

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| AI / LLM | Claude API (Anthropic) |
| Pipeline Architecture | LangGraph-style state machine |
| Tool Pattern | LangChain-style DataTools class |
| Data Processing | Pandas, NumPy |
| Visualizations | Matplotlib, Seaborn |
| Database | SQLite (in-memory execution) |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
DataMind/
├── main.py              # FastAPI backend + LangGraph pipeline + DataTools
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── file_meta.json       # File metadata (auto-created)
├── static/
│   └── index.html       # Frontend dashboard UI (6 tabs)
├── uploads/             # Uploaded datasets (auto-created)
└── venv/                # Virtual environment (exclude from GitHub)
```

---

## How to Run

**1. Create and activate virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies**
```powershell
pip install --prefer-binary -r requirements.txt
```

**3. Set your API key**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**4. Start the server**
```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**5. Open in browser**
```
http://localhost:8000
```

---

## Key Features

-  Automated data analysis — no manual steps needed
-  Multi-agent LangGraph-style pipeline
-  AI-generated charts and written report
-  SQL-based insight queries (auto-planned + manual editor)
-  Interactive 6-tab dashboard
-  Supports CSV, Excel (.xlsx), and SQLite (.db)
-  Click any chart to zoom fullscreen

---

## Sample Files to Test

| File | Description |
|---|---|
| `sales_data.csv` | 200 rows — product sales by category, region, date |
| `employee_data.xlsx` | 100 rows — salary, department, performance, experience |
| `ecommerce_store.db` | 3 tables — products, customers, orders (200 rows) |

---

## Use Case

Helps analysts and business users quickly understand any dataset without writing code. Upload a file, click **Run Analysis Agent**, and get a full dashboard with charts, statistics, SQL insights, and a written report in under 60 seconds.
