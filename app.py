import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="AI Agent Evaluation Lab",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Agent Evaluation Lab")
st.caption("Local LLM Benchmark & Performance Dashboard")

path = Path("reports/benchmark_results.csv")

if not path.exists():
    st.warning("No benchmark results found.")
    st.stop()

df = pd.read_csv(path)

col1, col2, col3 = st.columns(3)

col1.metric("AI Models", df["model"].nunique())
col2.metric("Total Tests", len(df))
col3.metric("Total Tokens", int(df["tokens"].sum()))

st.divider()

st.subheader("Model Performance")

summary = df.groupby("model").agg(
    avg_latency=("latency_seconds", "mean"),
    total_tokens=("tokens", "sum"),
    total_tests=("model", "count")
).round(2)

st.dataframe(summary)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Latency (s)")
    st.bar_chart(summary["avg_latency"])

with col2:
    st.subheader("Token Usage")
    st.bar_chart(summary["total_tokens"])

st.subheader("Detailed Benchmark Results")
st.dataframe(df)

st.download_button(
    "Download Benchmark CSV",
    data=path.read_bytes(),
    file_name="benchmark_results.csv",
    mime="text/csv"
)
st.divider()
st.header("AI Accuracy Evaluation")

accuracy_path = Path("reports/accuracy_results.csv")

if accuracy_path.exists():
    accuracy_df = pd.read_csv(accuracy_path)

    accuracy_summary = (
        accuracy_df.groupby("model")["accuracy_score"]
        .mean()
        .round(2)
    )

    cols = st.columns(len(accuracy_summary))

    for col, (model, score) in zip(
        cols, accuracy_summary.items()
    ):
        col.metric(model, f"{score:.1f}%")

    st.bar_chart(accuracy_summary)

    st.subheader("Detailed Accuracy Results")
    st.dataframe(accuracy_df)

    st.download_button(
        "Download Accuracy CSV",
        data=accuracy_path.read_bytes(),
        file_name="accuracy_results.csv",
        mime="text/csv"
    )
else:
    st.warning("No accuracy results found.")
    st.divider()
st.header("RAG Evaluation")

rag_path = Path("reports/rag_results.csv")

if rag_path.exists():
    rag_df = pd.read_csv(rag_path)

    rag_summary = (
        rag_df.groupby("model")["accuracy_score"]
        .mean()
        .round(2)
    )

    cols = st.columns(len(rag_summary))

    for col, (model, score) in zip(
        cols, rag_summary.items()
    ):
        col.metric(model, f"{score:.1f}%")

    st.bar_chart(rag_summary)

    st.subheader("Detailed RAG Results")

    st.dataframe(rag_df[[
        "model",
        "question",
        "response",
        "expected",
        "sources",
        "exact_score",
        "accuracy_score",
        "latency_seconds"
    ]])

    st.download_button(
        "Download RAG Results",
        data=rag_path.read_bytes(),
        file_name="rag_results.csv",
        mime="text/csv"
    )

else:
    st.warning("No RAG results found.")
    st.divider()
st.header("Tool Calling Evaluation")

tool_path = Path("reports/tool_results.csv")

if tool_path.exists():
    tool_df = pd.read_csv(tool_path)

    tool_summary = (
        tool_df.groupby("model")["tool_score"]
        .mean()
        .round(2)
    )

    cols = st.columns(len(tool_summary))

    for col, (model, score) in zip(
        cols, tool_summary.items()
    ):
        col.metric(model, f"{score:.1f}%")

    st.bar_chart(tool_summary)

    st.subheader("Detailed Tool Calling Results")
    st.dataframe(tool_df)

    st.download_button(
        "Download Tool Calling CSV",
        data=tool_path.read_bytes(),
        file_name="tool_results.csv",
        mime="text/csv"
    )
else:
    st.warning("No tool calling results found.")
