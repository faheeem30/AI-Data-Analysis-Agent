
# DataMind — AI Data Analysis Agent

An AI-powered data analysis system that automatically explores datasets, generates insights, creates visualizations, runs SQL queries, and produces a structured report — all in one click.

---

##  Overview

DataMind automates the complete data analysis workflow. Users can upload a dataset (CSV, Excel, SQLite), and the system analyzes it using a multi-step AI pipeline — delivering charts, statistics, SQL insights, and a final report.

---

## Application Screens

### Overview Dashboard
<img width="1907" height="964" alt="image" src="https://github.com/user-attachments/assets/eed64c37-7e96-4924-bca6-b0f4e4248e5d" />

Displays dataset KPIs like rows, columns, missing values, duplicates, and averages.

### Charts
<img width="1910" height="975" alt="image" src="https://github.com/user-attachments/assets/34766590-d30b-43d1-9f9b-59deaa11ca5d" />

- Histogram  
- Bar Charts  
- Line Charts  
- Scatter Plots  
- Heatmaps  

### AI Report
<img width="1911" height="975" alt="image" src="https://github.com/user-attachments/assets/5862abd1-e8c1-4bb5-bc87-7d4cdabe387c" />

- Correlation insights  
- Distribution analysis  
- Performance insights  
- Business recommendations  

###  SQL Results
<img width="1911" height="969" alt="image" src="https://github.com/user-attachments/assets/9b2caa86-fda2-4f35-acc7-d4222becab27" />
- Top products  
- Regional performance  
- Discount analysis  

### Stats
<img width="1911" height="926" alt="image" src="https://github.com/user-attachments/assets/118dec4a-e162-4497-8f56-ca46868dd794" />
- Column types  
- Missing values  
- Summary statistics  

###  SQL Editor
<img width="1915" height="933" alt="image" src="https://github.com/user-attachments/assets/b7372185-8764-48da-8cf6-8aa0f06f4721" />
Run custom SQL queries on uploaded data.

---

## How It Works

1. Upload dataset  
2. AI generates analysis plan  
3. Pipeline executes (Planner → Stats → Charts → SQL → Reporter)  
4. Outputs shown in dashboard  

---

## Tech Stack

- **Backend:** FastAPI  
- **AI:** Claude API  
- **Architecture:** LangGraph-style pipeline  
- **Tools:** LangChain-style tools  
- **Data:** Pandas, NumPy, SciPy  
- **Charts:** Matplotlib, Seaborn  
- **Database:** SQLite (in-memory)  
- **Frontend:** HTML, CSS, JavaScript  

---

## Project Structure

```

DataMind/
│── main.py                # FastAPI backend (core logic)
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── file_meta.json        # Metadata storage
│
├── static/
│   └── index.html        # Frontend UI
│
├── uploads/              # Uploaded datasets (auto-created)
├── outputs/              # Generated outputs (charts/reports if used)
├── **pycache**/          # Python cache files
├── venv/                 # Virtual environment (not needed in GitHub)

```

---

## How to Run

### 1. Install dependencies
```
python -m venv venv
venv\Scripts\activate

pip install --prefer-binary -r requirements.txt

```

### 2. Import the API keys
```

$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"


```


### 3. Run the server
```

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

### 4. Open in browser
```

[http://localhost:8000](http://localhost:8000)

```

---


---

## Key Features

- Automated data analysis (no manual steps)  
- Multi-agent pipeline  
- AI-generated insights  
- SQL-based analysis  
- Interactive dashboard  
- Supports CSV, Excel, SQLite  

---

## Use Case

Helps analysts and business users quickly understand datasets without writing code.

---

