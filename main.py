from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sqlite3
import anthropic
import os, shutil, json, re, base64, io, traceback
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI(title="DataMind — AI Data Analysis Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "./uploads"
META_FILE  = "./file_meta.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = anthropic.Anthropic()

# ── Styling ────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#1e2230",
    "axes.facecolor":   "#161920",
    "axes.edgecolor":   "#3a3f55",
    "axes.labelcolor":  "#e8eaf0",
    "xtick.color":      "#7a7f96",
    "ytick.color":      "#7a7f96",
    "text.color":       "#e8eaf0",
    "grid.color":       "#2a2f45",
    "grid.alpha":       0.5,
})
COLORS = ["#7c6dfa","#3ecf8e","#fb923c","#f87171","#60a5fa","#c084fc","#fbbf24","#34d399"]

# ══════════════════════════════════════════════════════════
# ── LangChain-style TOOLS ─────────────────────────────────
# ══════════════════════════════════════════════════════════
class DataTools:
    """LangChain-style tool collection for data analysis."""

    def __init__(self, df: pd.DataFrame, filename: str):
        self.df = df
        self.filename = filename
        self.charts = []   # base64 encoded charts
        self.insights = [] # text insights

    # Tool 1 — Basic stats
    def get_basic_stats(self) -> dict:
        """Get shape, dtypes, missing values, basic describe."""
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()
        missing      = df.isnull().sum()
        missing_pct  = (missing / len(df) * 100).round(2)

        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "missing_values": {col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                               for col in df.columns if missing[col] > 0},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        }
        if numeric_cols:
            desc = df[numeric_cols].describe().round(3)
            stats["numeric_summary"] = desc.to_dict()
        return stats

    # Tool 2 — Correlation
    def get_correlations(self) -> dict:
        """Compute correlation matrix for numeric columns."""
        numeric = self.df.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2:
            return {"error": "Not enough numeric columns for correlation"}
        corr = numeric.corr().round(3)
        # Find top correlations
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append({"col1": cols[i], "col2": cols[j], "r": round(corr.iloc[i,j], 3)})
        pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        return {"matrix": corr.to_dict(), "top_pairs": pairs[:5]}

    # Tool 3 — Value counts for categoricals
    def get_value_counts(self, col: str, top_n: int = 10) -> dict:
        """Get value counts for a categorical column."""
        if col not in self.df.columns:
            return {"error": f"Column '{col}' not found"}
        vc = self.df[col].value_counts().head(top_n)
        return {"column": col, "counts": vc.to_dict(), "unique": int(self.df[col].nunique())}

    # Tool 4 — Run SQL on the dataframe
    def run_sql(self, query: str) -> dict:
        """Run SQL query on the dataframe using sqlite3."""
        try:
            conn = sqlite3.connect(":memory:")
            tname = re.sub(r'\W+','_', os.path.splitext(self.filename)[0])
            self.df.to_sql(tname, conn, if_exists="replace", index=False)
            result = pd.read_sql_query(query, conn)
            conn.close()
            return {"rows": result.to_dict(orient="records"), "columns": list(result.columns)}
        except Exception as e:
            return {"error": str(e)}

    # Tool 5 — Chart: histogram
    def chart_histogram(self, col: str, title: str = "") -> str:
        """Generate histogram for a numeric column. Returns base64 PNG."""
        if col not in self.df.columns:
            return ""
        fig, ax = plt.subplots(figsize=(10, 5))
        data = self.df[col].dropna()
        ax.hist(data, bins=35, color=COLORS[0], alpha=0.85, edgecolor="#2a2f45")
        ax.set_title(title or f"Distribution of {col}", fontsize=15, pad=12, color="#e8eaf0")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.axvline(data.mean(), color=COLORS[1], linestyle="--", linewidth=2,
                   label=f"Mean: {data.mean():.2f}")
        ax.axvline(data.median(), color=COLORS[2], linestyle=":", linewidth=2,
                   label=f"Median: {data.median():.2f}")
        ax.legend(fontsize=10)
        plt.tight_layout()
        b64 = self._fig_to_b64(fig)
        plt.close(fig)
        self.charts.append({"type": "histogram", "col": col, "img": b64})
        return b64

    # Tool 6 — Chart: bar chart
    def chart_bar(self, col: str, value_col: str = None, top_n: int = 10, title: str = "") -> str:
        """Bar chart for categorical column (optionally aggregated by value_col)."""
        if col not in self.df.columns:
            return ""
        fig, ax = plt.subplots(figsize=(10, 5))
        if value_col and value_col in self.df.columns:
            data = self.df.groupby(col)[value_col].sum().nlargest(top_n)
            ylabel = f"Sum of {value_col}"
        else:
            data = self.df[col].value_counts().head(top_n)
            ylabel = "Count"
        bars = ax.bar(range(len(data)), data.values,
                      color=[COLORS[i % len(COLORS)] for i in range(len(data))],
                      alpha=0.88, edgecolor="#2a2f45", width=0.65)
        # value labels on bars
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(data.values)*0.01,
                    f"{val:,.0f}", ha='center', va='bottom', fontsize=9, color="#e8eaf0")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([str(x) for x in data.index], rotation=30, ha="right", fontsize=10)
        ax.set_title(title or f"{col} — Top {len(data)}", fontsize=15, pad=12)
        ax.set_ylabel(ylabel, fontsize=11)
        plt.tight_layout()
        b64 = self._fig_to_b64(fig)
        plt.close(fig)
        self.charts.append({"type": "bar", "col": col, "img": b64})
        return b64

    # Tool 7 — Chart: correlation heatmap
    def chart_heatmap(self) -> str:
        """Correlation heatmap for all numeric columns."""
        numeric = self.df.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2:
            return ""
        fig, ax = plt.subplots(figsize=(11, 8))
        corr = numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    annot=True, fmt=".2f", ax=ax,
                    linewidths=0.5, linecolor="#2a2f45",
                    annot_kws={"size": 10},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap", fontsize=15, pad=14)
        plt.tight_layout()
        b64 = self._fig_to_b64(fig)
        plt.close(fig)
        self.charts.append({"type": "heatmap", "img": b64})
        return b64

    # Tool 8 — Chart: line/time series
    def chart_line(self, x_col: str, y_col: str, title: str = "") -> str:
        """Line chart for time series or ordered data."""
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return ""
        fig, ax = plt.subplots(figsize=(12, 5))
        df_sorted = self.df[[x_col, y_col]].dropna().sort_values(x_col)
        x_vals = range(len(df_sorted))
        y_vals = df_sorted[y_col].values
        ax.plot(x_vals, y_vals, color=COLORS[0], linewidth=2, alpha=0.9)
        ax.fill_between(x_vals, y_vals, alpha=0.12, color=COLORS[0])
        # rolling average
        if len(df_sorted) > 10:
            import pandas as pd
            roll = pd.Series(y_vals).rolling(7, min_periods=1).mean()
            ax.plot(x_vals, roll, color=COLORS[1], linewidth=2,
                    linestyle="--", alpha=0.8, label="7-period avg")
            ax.legend(fontsize=10)
        step = max(1, len(df_sorted) // 10)
        ax.set_xticks(range(0, len(df_sorted), step))
        ax.set_xticklabels([str(v) for v in df_sorted[x_col].values[::step]],
                           rotation=30, ha="right", fontsize=9)
        ax.set_title(title or f"{y_col} over {x_col}", fontsize=15, pad=12)
        ax.set_ylabel(y_col, fontsize=11)
        plt.tight_layout()
        b64 = self._fig_to_b64(fig)
        plt.close(fig)
        self.charts.append({"type": "line", "x": x_col, "y": y_col, "img": b64})
        return b64

    # Tool 9 — Chart: scatter
    def chart_scatter(self, x_col: str, y_col: str, hue_col: str = None, title: str = "") -> str:
        """Scatter plot for two numeric columns."""
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        if hue_col and hue_col in self.df.columns:
            cats = self.df[hue_col].unique()
            for i, cat in enumerate(cats[:8]):
                mask = self.df[hue_col] == cat
                ax.scatter(self.df.loc[mask, x_col], self.df.loc[mask, y_col],
                           color=COLORS[i % len(COLORS)], alpha=0.7, s=45,
                           label=str(cat), edgecolors="#2a2f45", linewidths=0.4)
            ax.legend(fontsize=9, title=hue_col, title_fontsize=10)
        else:
            ax.scatter(self.df[x_col], self.df[y_col],
                       color=COLORS[0], alpha=0.65, s=45,
                       edgecolors="#2a2f45", linewidths=0.4)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(title or f"{x_col} vs {y_col}", fontsize=15, pad=12)
        plt.tight_layout()
        b64 = self._fig_to_b64(fig)
        plt.close(fig)
        self.charts.append({"type": "scatter", "x": x_col, "y": y_col, "img": b64})
        return b64

    def _fig_to_b64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════
# ── LangGraph-style STATE MACHINE ─────────────────────────
# ══════════════════════════════════════════════════════════
class AgentState:
    """Represents the state passed between LangGraph nodes."""
    def __init__(self, df: pd.DataFrame, filename: str, user_question: str = ""):
        self.df            = df
        self.filename      = filename
        self.user_question = user_question
        self.tools         = DataTools(df, filename)
        self.stats         = {}
        self.correlations  = {}
        self.plan          = []       # Claude's analysis plan
        self.executed      = []       # completed steps
        self.charts        = []       # generated charts
        self.sql_results   = []       # SQL query results
        self.report        = ""       # final written report
        self.kpis          = {}       # key metrics for dashboard
        self.errors        = []


# ── NODE 1: Planner ────────────────────────────────────────
def node_planner(state: AgentState) -> AgentState:
    """LangGraph node: Claude reads schema and creates an analysis plan."""
    df = state.df
    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols     = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
    sample        = df.head(3).to_string()

    prompt = f"""You are a data analysis agent. You will create a structured analysis plan for a dataset.

Dataset: {state.filename}
Shape: {len(df)} rows × {len(df.columns)} columns
Columns: {list(df.columns)}
Numeric columns: {numeric_cols}
Categorical columns: {cat_cols}
Date/time columns: {date_cols}
Sample rows:
{sample}

User question: {state.user_question or "Perform a comprehensive analysis"}

Create a JSON analysis plan with exactly this structure:
{{
  "title": "Analysis title",
  "objective": "What this analysis will uncover",
  "steps": [
    {{"id": 1, "name": "step name", "tool": "tool_name", "params": {{}}, "reason": "why"}},
    ...
  ],
  "sql_queries": [
    {{"name": "query name", "sql": "SELECT ...", "reason": "insight this gives"}}
  ],
  "kpi_columns": ["col1", "col2"]
}}

Available tools: get_basic_stats, chart_histogram(col), chart_bar(col), chart_heatmap, chart_line(x_col,y_col), chart_scatter(x_col,y_col), get_correlations, get_value_counts(col)

Rules:
- Plan 4-7 analysis steps based on what columns exist
- Include 2-3 relevant SQL queries
- Choose charts that make sense for the data types
- For numeric data: histogram + heatmap if >2 numeric cols
- For categorical data: bar charts
- For time data: line chart
- Output ONLY valid JSON, no explanation"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    # Extract JSON
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            plan_data = json.loads(m.group())
            state.plan = plan_data
        except:
            state.errors.append("Plan parsing failed")
    return state


# ── NODE 2: Stats Executor ─────────────────────────────────
def node_stats(state: AgentState) -> AgentState:
    """LangGraph node: Run basic stats and correlations."""
    state.stats        = state.tools.get_basic_stats()
    state.correlations = state.tools.get_correlations()

    # Build KPIs
    df = state.df
    numeric = df.select_dtypes(include=[np.number])
    kpis = {
        "total_rows":    len(df),
        "total_columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct":   round(df.isnull().sum().sum() / (len(df)*len(df.columns)) * 100, 1),
        "numeric_cols":  len(numeric.columns),
        "cat_cols":      len(df.select_dtypes(include=["object","category"]).columns),
        "duplicates":    int(df.duplicated().sum()),
    }
    if len(numeric.columns):
        for col in list(numeric.columns)[:3]:
            kpis[f"{col}_mean"] = round(float(numeric[col].mean()), 2)
            kpis[f"{col}_max"]  = round(float(numeric[col].max()),  2)
    state.kpis = kpis
    state.executed.append("stats")
    return state


# ── NODE 3: Chart Executor ─────────────────────────────────
def node_charts(state: AgentState) -> AgentState:
    """LangGraph node: Execute all planned chart steps."""
    tools = state.tools
    df    = state.df
    plan  = state.plan

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols    = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

    steps = plan.get("steps", []) if isinstance(plan, dict) else []

    for step in steps:
        try:
            tool   = step.get("tool", "")
            params = step.get("params", {})

            if tool == "chart_histogram":
                col = params.get("col") or (numeric_cols[0] if numeric_cols else None)
                if col and col in df.columns:
                    tools.chart_histogram(col, f"Distribution: {col}")

            elif tool == "chart_bar":
                col = params.get("col") or (cat_cols[0] if cat_cols else None)
                val = params.get("value_col")
                if col and col in df.columns:
                    tools.chart_bar(col, val if val and val in df.columns else None,
                                    title=f"Top values: {col}")

            elif tool == "chart_heatmap":
                if len(numeric_cols) >= 2:
                    tools.chart_heatmap()

            elif tool == "chart_line":
                x = params.get("x_col") or (date_cols[0] if date_cols else None)
                y = params.get("y_col") or (numeric_cols[0] if numeric_cols else None)
                if x and y and x in df.columns and y in df.columns:
                    tools.chart_line(x, y)

            elif tool == "chart_scatter":
                x = params.get("x_col") or (numeric_cols[0] if len(numeric_cols)>1 else None)
                y = params.get("y_col") or (numeric_cols[1] if len(numeric_cols)>1 else None)
                h = params.get("hue_col")
                if x and y and x in df.columns and y in df.columns:
                    tools.chart_scatter(x, y, h if h and h in df.columns else None)

        except Exception as e:
            state.errors.append(f"Chart error ({step.get('tool')}): {e}")

    # Fallback: always generate at least basic charts
    if len(tools.charts) == 0:
        if numeric_cols:
            tools.chart_histogram(numeric_cols[0])
        if cat_cols:
            tools.chart_bar(cat_cols[0])
        if len(numeric_cols) >= 2:
            tools.chart_heatmap()

    state.charts = tools.charts
    state.executed.append("charts")
    return state


# ── NODE 4: SQL Executor ───────────────────────────────────
def node_sql(state: AgentState) -> AgentState:
    """LangGraph node: Run planned SQL queries."""
    plan    = state.plan
    queries = plan.get("sql_queries", []) if isinstance(plan, dict) else []
    results = []

    for q in queries[:4]:
        try:
            sql  = q.get("sql", "")
            name = q.get("name", "Query")
            if sql:
                res = state.tools.run_sql(sql)
                if "error" not in res:
                    results.append({"name": name, "sql": sql,
                                    "reason": q.get("reason",""),
                                    "rows":   res["rows"][:20],
                                    "columns":res["columns"]})
        except Exception as e:
            state.errors.append(f"SQL error: {e}")

    state.sql_results = results
    state.executed.append("sql")
    return state


# ── NODE 5: Reporter ───────────────────────────────────────
def node_reporter(state: AgentState) -> AgentState:
    """LangGraph node: Claude writes the final analysis report."""
    stats   = state.stats
    corr    = state.correlations
    plan    = state.plan
    kpis    = state.kpis
    sql_res = state.sql_results

    sql_summary = "\n".join([
        f"- {r['name']}: {r['rows'][:3]}" for r in sql_res
    ]) if sql_res else "No SQL results"

    top_corr = corr.get("top_pairs", [])[:3] if isinstance(corr, dict) else []

    prompt = f"""You are a senior data analyst. Write a comprehensive analysis report.

Dataset: {state.filename}
Rows: {kpis.get('total_rows')} | Columns: {kpis.get('total_columns')}
Missing data: {kpis.get('missing_pct')}%
Duplicates: {kpis.get('duplicates')}

Numeric columns: {stats.get('numeric_columns',[])}
Categorical columns: {stats.get('categorical_columns',[])}

Top correlations: {top_corr}

SQL query insights:
{sql_summary}

Analysis plan objective: {plan.get('objective','') if isinstance(plan,dict) else ''}
User question: {state.user_question or 'General analysis'}

Write a professional report with these sections:
1. **Executive Summary** (3-4 sentences)
2. **Key Findings** (5-7 bullet points with specific numbers)
3. **Data Quality** (missing values, duplicates, concerns)
4. **Statistical Insights** (correlations, distributions, outliers)
5. **Recommendations** (3-4 actionable recommendations)

Be specific, use numbers from the data. Write in markdown format."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    state.report = response.content[0].text
    state.executed.append("report")
    return state


# ══════════════════════════════════════════════════════════
# ── LangGraph RUNNER ──────────────────────────────────────
# ══════════════════════════════════════════════════════════
def run_graph(df: pd.DataFrame, filename: str, question: str = "") -> dict:
    """
    LangGraph-style execution:
    planner → stats → charts → sql → reporter
    Each node reads + writes AgentState.
    """
    state = AgentState(df, filename, question)

    # Execute nodes in order (graph edges)
    nodes = [
        ("planner",  node_planner),
        ("stats",    node_stats),
        ("charts",   node_charts),
        ("sql",      node_sql),
        ("reporter", node_reporter),
    ]

    node_log = []
    for name, fn in nodes:
        try:
            state = fn(state)
            node_log.append({"node": name, "status": "ok"})
        except Exception as e:
            state.errors.append(f"Node {name} failed: {e}")
            node_log.append({"node": name, "status": "error", "msg": str(e)})

    return {
        "filename":    state.filename,
        "kpis":        state.kpis,
        "stats":       state.stats,
        "plan":        state.plan,
        "charts":      state.charts,
        "sql_results": state.sql_results,
        "report":      state.report,
        "node_log":    node_log,
        "errors":      state.errors,
    }


# ══════════════════════════════════════════════════════════
# ── HELPERS ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════
def load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE) as f: return json.load(f)
    return {}

def save_meta(m: dict):
    with open(META_FILE,"w") as f: json.dump(m, f)

def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx",".xls"):
        return pd.read_excel(path)
    elif ext in (".db",".sqlite",".sqlite3"):
        conn = sqlite3.connect(path)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if not tables: raise ValueError("No tables found in database")
        df = pd.read_sql_query(f"SELECT * FROM \"{tables[0]}\"", conn)
        conn.close()
        return df
    raise ValueError(f"Unsupported file type: {ext}")


# ══════════════════════════════════════════════════════════
# ── API ENDPOINTS ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    filename: str
    question: str = ""

class SQLRequest(BaseModel):
    filename: str
    sql: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = (".csv",".xlsx",".xls",".db",".sqlite",".sqlite3")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Allowed: {', '.join(allowed)}")
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path,"wb") as f: shutil.copyfileobj(file.file, f)
    try:
        df   = load_file(path)
        cols = list(df.columns)
        meta = load_meta()
        meta[file.filename] = {
            "rows": len(df), "columns": len(df.columns),
            "col_names": cols,
            "dtypes": {c: str(df[c].dtype) for c in cols},
            "preview": df.head(3).fillna("").to_dict(orient="records"),
        }
        save_meta(meta)
        return {"filename": file.filename, "rows": len(df),
                "columns": cols, "preview": meta[file.filename]["preview"]}
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")


@app.get("/files")
async def list_files():
    return load_meta()


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    meta = load_meta()
    if filename not in meta:
        raise HTTPException(404, "Not found")
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path): os.remove(path)
    del meta[filename]; save_meta(meta)
    return {"message": f"Deleted '{filename}'"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    path = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found. Please upload it first.")
    try:
        df     = load_file(path)
        result = run_graph(df, req.filename, req.question)
        return result
    except Exception as e:
        raise HTTPException(500, f"Analysis error: {traceback.format_exc()}")


@app.post("/quick-sql")
async def quick_sql(req: SQLRequest):
    path = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    try:
        df   = load_file(path)
        conn = sqlite3.connect(":memory:")
        tname = re.sub(r'\W+','_', os.path.splitext(req.filename)[0])
        df.to_sql(tname, conn, if_exists="replace", index=False)
        result = pd.read_sql_query(req.sql, conn)
        conn.close()
        return {"rows": result.fillna("").to_dict(orient="records"),
                "columns": list(result.columns), "count": len(result)}
    except Exception as e:
        raise HTTPException(400, f"SQL Error: {e}")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
