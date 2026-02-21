"""
app.py
------
SMART-QA Lite – Rejection Reduction Analytics Dashboard
Streamlit entry point. Run with:
    streamlit run app.py
"""

import io
import warnings
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Local modules ──────────────────────────────────────────────────────────────
from data_loader     import load_data, get_data_summary
from analytics       import (
    compute_rejection_summary, plot_rejection_trends,
    compute_pareto,            plot_pareto,
    compute_correlation,       plot_correlation_heatmap,
    run_regression,            plot_regression_results,
)
from spc             import compute_pchart, plot_pchart, get_ooc_summary
from recommendation  import (
    generate_recommendations, recommendations_to_df, get_summary_counts,
    PRIORITY_EMOJI,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "SMART-QA Lite",
    page_icon   = "🏭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Top header bar */
    .main-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { color: rgba(255,255,255,0.82); margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #1B4F72;
    }
    .kpi-card.red   { border-left-color: #E74C3C; }
    .kpi-card.amber { border-left-color: #F39C12; }
    .kpi-card.green { border-left-color: #27AE60; }

    .kpi-value { font-size: 2rem; font-weight: 700; color: #1B4F72; }
    .kpi-label { font-size: 0.78rem; color: #7F8C8D; text-transform: uppercase;
                 letter-spacing: 0.06em; margin-top: 0.3rem; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1B4F72;
        border-bottom: 2px solid #AED6F1; padding-bottom: 0.35rem;
        margin: 1.5rem 0 1rem;
    }

    /* Recommendation cards */
    .rec-HIGH   { background:#FDEDEC; border-left:4px solid #E74C3C; border-radius:8px;
                  padding:0.8rem 1rem; margin-bottom:0.6rem; }
    .rec-MEDIUM { background:#FEFAE3; border-left:4px solid #F39C12; border-radius:8px;
                  padding:0.8rem 1rem; margin-bottom:0.6rem; }
    .rec-LOW    { background:#EAFAF1; border-left:4px solid #27AE60; border-radius:8px;
                  padding:0.8rem 1rem; margin-bottom:0.6rem; }
    .rec-INFO   { background:#EBF5FB; border-left:4px solid #2E86C1; border-radius:8px;
                  padding:0.8rem 1rem; margin-bottom:0.6rem; }

    .rec-subject { font-weight: 700; font-size: 0.97rem; }
    .rec-finding { font-size: 0.88rem; color: #555; margin-top: 0.2rem; }
    .rec-action  { font-size: 0.85rem; color: #2C3E50; margin-top: 0.4rem;
                   white-space: pre-line; }
    .rec-metric  { font-size: 0.78rem; color: #999; margin-top: 0.3rem;
                   font-family: 'DM Mono', monospace; }

    /* Info boxes */
    .info-box {
        background: #EBF5FB; border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #2C3E50; margin-top: 0.5rem;
    }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_to_streamlit(fig: plt.Figure):
    """Render matplotlib figure in Streamlit without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def kpi_card(value, label, color_class=""):
    st.markdown(
        f"""<div class="kpi-card {color_class}">
               <div class="kpi-value">{value}</div>
               <div class="kpi-label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_recommendation(rec):
    st.markdown(
        f"""<div class="rec-{rec.priority}">
               <div class="rec-subject">{PRIORITY_EMOJI[rec.priority]} [{rec.category}] {rec.subject}</div>
               <div class="rec-finding">📋 {rec.finding}</div>
               <div class="rec-action">✅ {rec.action}</div>
               {"<div class='rec-metric'>📊 " + rec.metric + "</div>" if rec.metric else ""}
            </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=60)
    st.title("SMART-QA Lite")
    st.caption("Rejection Reduction Analytics Framework")
    st.divider()

    uploaded = st.file_uploader(
        "📂 Upload Production Data",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file with the required columns.",
    )

    use_demo = st.checkbox("✅ Use built-in demo dataset", value=True)
    st.divider()

    st.markdown("**📌 Required Columns**")
    st.code(
        "Date, Shift, Machine_ID, Operator_ID,\n"
        "Produced_Qty, Rejected_Qty, Defect_Type,\n"
        "Material_Batch, Temperature, Pressure, Speed",
        language="text",
    )
    st.divider()
    st.markdown("**🛠 About**")
    st.caption(
        "A low-cost quality analytics system for small "
        "manufacturing industries. Built with Python + Streamlit."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>🏭 SMART-QA Lite – Rejection Reduction Analytics Framework</h1>
  <p>Statistical quality control · Pareto analysis · SPC monitoring · Actionable recommendations</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

df = None

if uploaded is not None:
    try:
        df = load_data(uploaded)
        st.success(f"✅ Data loaded successfully: **{len(df):,} rows**")
    except ValueError as e:
        st.error(f"❌ Data error: {e}")
        st.stop()
elif use_demo:
    try:
        df = load_data("sample_data.csv")
        st.info("🎯 Demo dataset loaded. Upload your own data in the sidebar to analyse real production data.")
    except Exception as e:
        st.error(f"Could not load demo data: {e}")
        st.stop()
else:
    st.markdown("""
    <div class="info-box">
    👆 Upload a CSV or Excel file using the sidebar, or check <b>"Use built-in demo dataset"</b> to explore the dashboard.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Running analytics…")
def run_all_analytics(df_json: str):
    """Cache-friendly wrapper — accepts JSON so Streamlit can hash it."""
    _df = pd.read_json(df_json, convert_dates=["Date"])

    summary   = compute_rejection_summary(_df)
    pareto    = compute_pareto(_df)
    corr_mat  = compute_correlation(_df)
    reg       = run_regression(_df)
    pc        = compute_pchart(_df, group_by="Date")
    recs      = generate_recommendations(_df, summary, pareto, corr_mat, pc, reg)
    data_sum  = get_data_summary(_df)

    return summary, pareto, corr_mat, reg, pc, recs, data_sum

with st.spinner("Crunching numbers…"):
    summary, pareto, corr_mat, reg, pc, recs, data_sum = run_all_analytics(
        df.to_json(date_format="iso")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

ovc1, ovc2, ovc3, ovc4, ovc5 = st.columns(5)

with ovc1:
    kpi_card(f"{data_sum['overall_rejection_rate']}%", "Overall Rejection Rate",
             "red" if data_sum["overall_rejection_rate"] > 8 else
             "amber" if data_sum["overall_rejection_rate"] > 5 else "green")

with ovc2:
    top_defect = pareto.iloc[0]["Defect_Type"]
    top_defect_pct = pareto.iloc[0]["Pct_Contribution"]
    kpi_card(f"{top_defect}", f"Top Defect ({top_defect_pct:.0f}%)", "red")

with ovc3:
    worst_machine = summary["machine"].iloc[0]
    kpi_card(f"{worst_machine['Machine_ID']}",
             f"Worst Machine ({worst_machine['Rejection_Rate_Pct']:.1f}%)", "amber")

with ovc4:
    ooc_count = int(pc["OOC"].sum())
    kpi_card(f"{ooc_count}", "OOC SPC Points",
             "red" if ooc_count >= 3 else "amber" if ooc_count >= 1 else "green")

with ovc5:
    kpi_card(f"{data_sum['total_produced']:,}", "Total Units Produced")


# Data overview
with st.expander("📋 Dataset Overview", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        - **Date Range:** {data_sum['date_range']}
        - **Total Rows:** {data_sum['total_rows']:,}
        - **Total Produced:** {data_sum['total_produced']:,}
        - **Total Rejected:** {data_sum['total_rejected']:,}
        """)
    with c2:
        st.markdown(f"""
        - **Machines:** {', '.join(data_sum['machines'])}
        - **Operators:** {', '.join(data_sum['operators'])}
        - **Defect Types:** {', '.join(data_sum['defect_types'])}
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Rejection Trends",
    "🏆 Pareto Analysis",
    "📉 SPC Chart",
    "🔗 Correlation & Regression",
    "💡 Recommendations",
])


# ── TAB 1: Rejection Trends ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Rejection Rate Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Formula:</b>  Rejection Rate (%) = (Rejected Qty / Produced Qty) × 100<br>
    Charts show daily trend, machine-wise, operator-wise, and weekly aggregation.
    </div>
    """, unsafe_allow_html=True)

    fig_trends = plot_rejection_trends(summary)
    fig_to_streamlit(fig_trends)

    # Detail tables
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Machine Summary**")
        st.dataframe(
            summary["machine"].style.format({"Rejection_Rate_Pct": "{:.2f}%"}),
            use_container_width=True, hide_index=True
        )
    with c2:
        st.markdown("**Operator Summary**")
        st.dataframe(
            summary["operator"].style.format({"Rejection_Rate_Pct": "{:.2f}%"}),
            use_container_width=True, hide_index=True
        )


# ── TAB 2: Pareto ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Pareto Analysis – 80/20 Rule</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    The Pareto Principle states that ~80% of defects come from ~20% of causes.
    <b>Red bars</b> = "Vital Few" defects to prioritise for maximum impact.
    </div>
    """, unsafe_allow_html=True)

    fig_pareto = plot_pareto(pareto)
    fig_to_streamlit(fig_pareto)

    st.markdown("**Defect Contribution Table**")
    st.dataframe(
        pareto.style.format({
            "Total_Rejected": "{:,}",
            "Pct_Contribution": "{:.1f}%",
            "Cumulative_Pct": "{:.1f}%",
        }).apply(
            lambda row: ["background-color: #FDEDEC" if row["Vital_Few"] else "" for _ in row],
            axis=1
        ),
        use_container_width=True, hide_index=True
    )


# ── TAB 3: SPC ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Statistical Process Control – p-Chart</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>p-Chart Logic:</b><br>
    • Centre Line (p̄) = Σ Rejected / Σ Produced<br>
    • UCL = p̄ + 3√(p̄(1−p̄)/n)&nbsp;&nbsp;&nbsp;&nbsp;LCL = max(0, p̄ − 3√(p̄(1−p̄)/n))<br>
    • <b>Red triangles</b> = out-of-control points requiring investigation
    </div>
    """, unsafe_allow_html=True)

    fig_spc = plot_pchart(pc)
    fig_to_streamlit(fig_spc)

    ooc_table = get_ooc_summary(pc)
    if len(ooc_table):
        st.markdown(f"**Out-of-Control Points ({len(ooc_table)} detected)**")
        st.dataframe(ooc_table, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No out-of-control points detected. Process is stable.")


# ── TAB 4: Correlation & Regression ───────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Correlation & Regression Analysis</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Correlation Heatmap**")
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem">
        Pearson r ranges from –1 (perfect negative) to +1 (perfect positive).
        Values |r| > 0.5 indicate strong influence on rejection rate.
        </div>
        """, unsafe_allow_html=True)
        fig_corr = plot_correlation_heatmap(corr_mat)
        fig_to_streamlit(fig_corr)

    with col_right:
        st.markdown("**Model Performance**")
        m1, m2 = st.columns(2)
        with m1:
            kpi_card(f"{reg['r2']}", "R² Score")
        with m2:
            kpi_card(f"{reg['rmse']}", "RMSE (%)")

        st.markdown("**Feature Coefficients (Standardised)**")
        st.dataframe(
            reg["coefficients"].style.format({
                "Coefficient": "{:.4f}",
                "Abs_Impact": "{:.4f}",
            }),
            use_container_width=True, hide_index=True
        )
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem">
        Positive coefficient → higher value increases rejection rate.<br>
        Negative coefficient → higher value reduces rejection rate.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Actual vs Predicted & Feature Importance**")
    fig_reg = plot_regression_results(reg)
    fig_to_streamlit(fig_reg)


# ── TAB 5: Recommendations ───────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">💡 Actionable Recommendations</div>', unsafe_allow_html=True)

    counts = get_summary_counts(recs)
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1: kpi_card(counts["HIGH"],   "🔴 High Priority",   "red")
    with rc2: kpi_card(counts["MEDIUM"], "🟡 Medium Priority", "amber")
    with rc3: kpi_card(counts["LOW"],    "🟢 Low Priority",    "green")
    with rc4: kpi_card(counts["INFO"],   "🔵 Info",            "")

    st.divider()

    # Filter controls
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_priority = st.multiselect(
            "Filter by Priority",
            options=["HIGH", "MEDIUM", "LOW", "INFO"],
            default=["HIGH", "MEDIUM"],
        )
    with col_f2:
        categories = list({r.category for r in recs})
        filter_cat = st.multiselect("Filter by Category", options=categories, default=categories)

    filtered = [r for r in recs if r.priority in filter_priority and r.category in filter_cat]

    if not filtered:
        st.info("No recommendations match the selected filters.")
    else:
        for rec in filtered:
            render_recommendation(rec)

    # Download table
    st.divider()
    rec_df = recommendations_to_df(recs)
    csv_bytes = rec_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Recommendations (CSV)",
        data    = csv_bytes,
        file_name = "smart_qa_recommendations.csv",
        mime    = "text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<center><small>🏭 SMART-QA Lite · Built with Python · Pandas · Scikit-learn · Streamlit"
    " · No paid software required</small></center>",
    unsafe_allow_html=True,
)
