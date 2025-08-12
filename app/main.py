# app/main.py
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Radiotherapy QA Dashboard", page_icon="📊", layout="wide")
sns.set_style("whitegrid")

st.title("📊 Radiotherapy QA Dashboard")
st.markdown("Monitor LINAC QA checks, see alerts, and get quick insights.")

# ----------------------------
# Paths & Safe load
# ----------------------------
# Use your absolute CSV path
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "qa_cleaned_with_alerts.csv")
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize column names (tolerant)
    df.columns = [c.strip() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in df.columns:
        if col.lower().startswith("deviation"):
            df.rename(columns={col: "Deviation(%)"}, inplace=True)
            df["Deviation(%)"] = pd.to_numeric(df["Deviation(%)"], errors="coerce")
            break
    return df

df = load_data(DATA_PATH)

if df.empty:
    st.warning(
        "⚠️ No processed data file found. Please check `qa_cleaned_with_alerts.csv` exists at the specified path."
    )
    st.info(f"Expected path: `{DATA_PATH}`")
    st.stop()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

min_date = df["Date"].min().date() if not df["Date"].isnull().all() else None
max_date = df["Date"].max().date() if not df["Date"].isnull().all() else None

date_range = st.sidebar.date_input("Date range", value=(min_date, max_date))
machine_selector = st.sidebar.multiselect(
    "Machine(s)", options=sorted(df["Machine_ID"].unique()), default=sorted(df["Machine_ID"].unique())
)
test_selector = st.sidebar.multiselect(
    "Test Type(s)", options=sorted(df["Test_Type"].unique()), default=sorted(df["Test_Type"].unique())
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
else:
    start_date, end_date = df["Date"].min(), df["Date"].max() + pd.Timedelta(days=1)

df_filtered = df[
    (df["Date"] >= start_date) & (df["Date"] < end_date) &
    (df["Machine_ID"].isin(machine_selector)) &
    (df["Test_Type"].isin(test_selector))
].copy()

# ----------------------------
# KPIs
# ----------------------------
total_checks = len(df_filtered)
total_fail = (df_filtered["Status"].str.lower() == "fail").sum() if "Status" in df_filtered.columns else 0
total_pass = (df_filtered["Status"].str.lower() == "pass").sum() if "Status" in df_filtered.columns else 0
pass_rate = round((total_pass / total_checks) * 100, 1) if total_checks else 0
fail_rate = round((total_fail / total_checks) * 100, 1) if total_checks else 0
machines_in_alert = df_filtered[df_filtered["Status"].str.lower() == "fail"]["Machine_ID"].nunique() if "Status" in df_filtered.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total QA Checks", f"{total_checks:,}")
col2.metric("Pass Rate", f"{pass_rate}%", delta=None)
col3.metric("Fail Rate", f"{fail_rate}%", delta=None)
col4.metric("Machines in Alert", f"{machines_in_alert}")

st.markdown("---")

# ----------------------------
# Alerts
# ----------------------------
fails_df = df_filtered[df_filtered["Status"].str.lower() == "fail"].sort_values("Date", ascending=False)

if len(fails_df) > 0:
    st.subheader("⚠️ Active Alerts")
    st.warning(f"{len(fails_df)} failed QA checks in the selected range.")
    st.dataframe(fails_df[["Date", "Machine_ID", "Test_Type", "Measured_Value", "Expected_Value", "Deviation(%)", "Status"]].head(20), use_container_width=True)
else:
    st.success("No failed QA checks within the selected filters. ✅")

# ----------------------------
# Quick Insight
# ----------------------------
st.markdown("#### Quick Insight")
overall_fail_rate = round((df[df["Status"].str.lower() == "fail"].shape[0] / df.shape[0]) * 100, 2) if df.shape[0] else 0
if fail_rate > overall_fail_rate:
    st.info(f"Failure rate for the current selection is **{fail_rate}%**, higher than the overall average of **{overall_fail_rate}%**.")
elif fail_rate < overall_fail_rate:
    st.info(f"Failure rate for the current selection is **{fail_rate}%**, lower than the overall average of **{overall_fail_rate}%**.")
else:
    st.info(f"Failure rate for the current selection is **{fail_rate}%**, matching the overall average of **{overall_fail_rate}%**.")

st.markdown("---")

# ----------------------------
# Failure summary
# ----------------------------
st.subheader("📋 Failure Summary: Machine × Test Type")
failure_summary = (
    df_filtered[df_filtered["Status"].str.lower() == "fail"]
    .groupby(["Machine_ID", "Test_Type"])
    .size()
    .reset_index(name="Failure_Count")
    .sort_values("Failure_Count", ascending=False)
)

if failure_summary.empty:
    st.write("No failures in the selected filters.")
else:
    st.dataframe(failure_summary, use_container_width=True)
    per_machine = df_filtered.groupby("Machine_ID").agg(
        total_checks=("Status", "count"),
        total_fails=("Status", lambda s: (s.str.lower() == "fail").sum())
    )
    per_machine["Fail_Rate(%)"] = (per_machine["total_fails"] / per_machine["total_checks"] * 100).round(2)
    top_machine = per_machine["Fail_Rate(%)"].idxmax()
    top_rate = per_machine.loc[top_machine, "Fail_Rate(%)"]
    avg_rate = per_machine["Fail_Rate(%)"].mean().round(2)
    st.markdown(f"**Insight:** `{top_machine}` has the highest failure rate at **{top_rate}%**, vs average **{avg_rate}%**.")

st.markdown("---")

# ----------------------------
# Top failing test types
# ----------------------------
st.subheader("🔬 Top Failing Test Types")
test_fail_counts = (
    df_filtered[df_filtered["Status"].str.lower() == "fail"]
    .groupby("Test_Type")
    .size()
    .reset_index(name="Failure_Count")
    .sort_values("Failure_Count", ascending=False)
)
if test_fail_counts.empty:
    st.write("No failing test types in the selected range.")
else:
    st.dataframe(test_fail_counts, use_container_width=True)
    top_test = test_fail_counts.iloc[0]["Test_Type"]
    st.markdown(f"**Insight:** Most failures are from **{top_test}** ({int(test_fail_counts.iloc[0]['Failure_Count'])} failures).")

st.markdown("---")

# ----------------------------
# Deviation trend plot
# ----------------------------
st.subheader("📈 Deviation Over Time (by Test Type)")
fig, ax = plt.subplots(figsize=(10, 4))
try:
    sns.lineplot(data=df_filtered, x="Date", y="Deviation(%)", hue="Test_Type", marker="o", ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.axhline(2, color="red", linestyle="--", linewidth=0.7)
    ax.axhline(-2, color="red", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Deviation (%)")
    ax.set_title("Deviation Over Time")
    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.error("Plot failed: " + str(e))

# ----------------------------
# Deviation trend insight
# ----------------------------
try:
    max_date_all = df_filtered["Date"].max()
    recent_start = max_date_all - pd.Timedelta(days=30)
    prev_start = recent_start - pd.Timedelta(days=30)

    recent_mean = df_filtered[(df_filtered["Date"] > recent_start)]["Deviation(%)"].mean()
    prev_mean = df_filtered[(df_filtered["Date"] > prev_start) & (df_filtered["Date"] <= recent_start)]["Deviation(%)"].mean()

    if pd.notna(recent_mean) and pd.notna(prev_mean):
        diff = recent_mean - prev_mean
        pct_change = (diff / abs(prev_mean) * 100) if prev_mean != 0 else np.nan
        if diff > 0:
            st.info(f"Average deviation increased by {diff:.2f}% ({pct_change:.1f}%) in the last 30 days vs previous 30 days.")
        elif diff < 0:
            st.info(f"Average deviation decreased by {abs(diff):.2f}% — performance improving.")
        else:
            st.info("No change in average deviation between the last two 30-day windows.")
    else:
        st.info("Not enough data to compute deviation trend.")
except Exception:
    st.info("Unable to compute deviation trend.")

st.markdown("---")

# ----------------------------
# Download & raw data
# ----------------------------
st.subheader("Export / Raw Data")
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(label="📥 Download filtered CSV", data=csv, file_name="qa_filtered.csv", mime="text/csv")

with st.expander("📄 View filtered rows"):
    st.dataframe(df_filtered, use_container_width=True)

st.markdown("---")
st.caption("Developed by Lenix Owino — Medical Physicist & Data Analyst")
