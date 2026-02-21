# 🏭 SMART-QA Lite – Rejection Reduction Analytics Framework

A low-cost, practical data analytics system for small manufacturing industries.  

No paid software · No IoT sensors · No advanced IT infrastructure required.

<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/d984ce56-18be-45fd-bd6e-e4ec5f8fe98c" />


## 📌 Project Overview

SMART-QA Lite helps manufacturing teams reduce rejection rates by turning raw
production data into clear, actionable insights — using only Python and a web
browser.

| Feature | Details |
|---|---|
| Input | CSV or Excel (drag & drop) |
| Analytics Engine | Python (Pandas, NumPy, Scikit-learn) |
| Dashboard | Streamlit (runs in browser) |
| Cost | ₹0 / $0 — 100% open-source |
| Skill Level Required | Basic (non-IT users can operate the dashboard) |

---

## 🗂 Project Structure

```
smart_qa_lite/
│
├── app.py               # Streamlit dashboard (entry point)
├── data_loader.py       # CSV/Excel ingestion & validation
├── analytics.py         # Rejection rate, Pareto, Correlation, Regression
├── spc.py               # Statistical Process Control (p-chart)
├── recommendation.py    # Rule-based recommendation engine
│
├── sample_data.csv      # Demo dataset (14 days, 3 machines, 6 operators)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                    Streamlit Dashboard                          │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  KPI Cards | Pareto | SPC Chart | Correlation | Recs     │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  File Upload / Sample Data
┌──────────────────────────▼──────────────────────────────────────┐
│                      app.py (Orchestrator)                      │
│   Calls: data_loader → analytics → spc → recommendation        │
└──┬───────────────┬──────────────┬──────────────┬───────────────┘
   │               │              │              │
┌──▼──────┐  ┌────▼─────┐  ┌────▼──────┐  ┌───▼──────────┐
│data_    │  │analytics │  │  spc.py   │  │recommendation│
│loader  │  │.py       │  │p-chart    │  │.py           │
│─────── │  │──────────│  │UCL/LCL    │  │Rule engine   │
│CSV read│  │Rejection │  │OOC detect │  │Machine rules │
│Validate│  │rate calc │  └───────────┘  │Operator rules│
│Clean   │  │Pareto    │                 │SPC rules     │
│Enrich  │  │Corr/Reg  │                 │Pareto rules  │
└────────┘  └──────────┘                 │Param rules   │
                                         └──────────────┘
```

---

## 📊 Analytics Modules

### 1️⃣ Rejection Rate Calculation
```
Rejection_Rate (%) = (Rejected_Qty / Produced_Qty) × 100
```
Generates: Daily trend · Machine-wise · Operator-wise · Weekly

### 2️⃣ Pareto Analysis (80/20 Rule)
Aggregates defect categories by total rejected count.
Highlights the "Vital Few" defects contributing ≤80% cumulatively.

### 3️⃣ Statistical Process Control (p-Chart)
```
p̄   = Σ d_i / Σ n_i
UCL = p̄ + 3√(p̄(1−p̄)/n)
LCL = max(0, p̄ − 3√(p̄(1−p̄)/n))
```
Detects: Points beyond limits · Run of 8 · Trend of 6

### 4️⃣ Correlation & Regression
- Pearson correlation matrix
- OLS regression: Rejection_Rate ~ Temperature + Pressure + Speed
- Standardised coefficients for interpretable feature importance

### 5️⃣ Recommendation Engine (Rule-Based)
| Trigger | Action |
|---|---|
| Machine rejection > 8% | Schedule maintenance |
| Operator rate > avg × 1.5 | Suggest retraining |
| OOC ≥ 2 SPC points | Investigate process stability |
| \|r\| > 0.5 with Rejection Rate | Control that parameter |
| Defect in Vital Few | Root-cause analysis |
| Batch rate > avg × 1.3 | Supplier quality concern |

---

## 🚀 Deployment Guide

### Step 1 – Install Python (3.9+)
Download from: https://www.python.org/downloads/

### Step 2 – Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 – Run the Dashboard
```bash
cd smart_qa_lite
streamlit run app.py
```
The dashboard opens at: http://localhost:8501

### Step 4 – Use Your Data
1. Prepare your data in CSV or Excel format with the required columns
2. Open the dashboard in your browser
3. Upload your file using the sidebar
4. All charts and recommendations update automatically

---

## 📋 Required CSV/Excel Columns

| Column | Type | Example |
|---|---|---|
| Date | Date | 2024-01-15 |
| Shift | Text | Morning / Afternoon / Night |
| Machine_ID | Text | M1, M2 |
| Operator_ID | Text | OP1, OP2 |
| Produced_Qty | Number | 500 |
| Rejected_Qty | Number | 25 |
| Defect_Type | Text | Porosity, Dimensional |
| Material_Batch | Text | B001 |
| Temperature | Number | 185.5 |
| Pressure | Number | 6.2 |
| Speed | Number | 120 |

---

## 🔮 Future Enhancements

1. **Shift-wise p-chart** for intra-day monitoring
2. **Logistic Regression** for binary rejection prediction
3. **Isolation Forest** for anomaly detection in process parameters
4. **Email alerts** when SPC triggers out-of-control signals
5. **PDF report generation** for management review
6. **Multi-plant comparison** dashboard
7. **Real-time mode** with periodic CSV refresh


*Built for small manufacturers. Practical, interpretable, zero-cost.*
