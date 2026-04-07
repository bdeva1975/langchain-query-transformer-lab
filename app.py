# app.py
# Streamlit UI for the LangChain Query Transformer Lab.
# Run with: streamlit run app.py

import os
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.transformers import TRANSFORMERS
from src.retriever import build_index, retrieve
from src.evaluator import evaluate

# ── Setup ──────────────────────────────────────────────────────────
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Query Transformer Lab",
    page_icon="🔬",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────
st.title("🔬 LangChain Query Transformer Lab")
st.markdown(
    "Compare **4 query transformation techniques** side by side — "
    "Rewrite, Multi-Query, Step-Back and HyDE.\n\n"
    "Drop your documents in the `data/` folder, type a query, "
    "and see how each transformer affects retrieval quality."
)
st.divider()

# ── Load Config ────────────────────────────────────────────────────
config = load_config()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model")
    config["llm"]["model"] = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )

    st.subheader("Retrieval")
    config["retrieval"]["top_k"] = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1, max_value=10, value=3
    )

    st.subheader("Transformers")
    selected_transformers = st.multiselect(
        "Select transformers to run",
        options=list(TRANSFORMERS.keys()),
        default=list(TRANSFORMERS.keys()),
        format_func=lambda x: {
            "baseline": "Baseline (No Transformation)",
            "rewrite": "Rewrite-Retrieve-Read",
            "multi_query": "Multi-Query Generation",
            "step_back": "Step-Back Questioning",
            "hyde": "HyDE"
        }.get(x, x)
    )

    st.divider()
    st.subheader("Index Documents")
    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building index from data/ folder..."):
            try:
                st.session_state["collection"] = build_index(config)
                st.success("✅ Index built successfully!")
            except Exception as e:
                st.error(f"❌ {e}")

    if "collection" in st.session_state:
        st.success("✅ Index ready")
    else:
        st.warning("⚠️ Index not built yet")

# ── Main Area ──────────────────────────────────────────────────────
st.subheader("💬 Enter Your Query")
query = st.text_input(
    "Query",
    placeholder="e.g. What are the main chunking strategies in RAG?",
    label_visibility="collapsed"
)

run_button = st.button(
    "🚀 Run Transformers",
    type="primary",
    use_container_width=False,
    disabled="collection" not in st.session_state or not query
)

# ── Run ────────────────────────────────────────────────────────────
if run_button and query:
    if "collection" not in st.session_state:
        st.error("Please build the index first using the sidebar.")
    else:
        collection = st.session_state["collection"]
        all_results = []

        st.divider()
        st.subheader("📊 Results")

        tabs = st.tabs([
            {
                "baseline": "🔵 Baseline",
                "rewrite": "✏️ Rewrite",
                "multi_query": "🔀 Multi-Query",
                "step_back": "⬆️ Step-Back",
                "hyde": "💡 HyDE"
            }.get(t, t)
            for t in selected_transformers
        ])

        for i, transformer_key in enumerate(selected_transformers):
            transformer_fn = TRANSFORMERS[transformer_key]

            with tabs[i]:
                with st.spinner(f"Running {transformer_key}..."):
                    # Transform
                    result = transformer_fn(query, config)
                    # Retrieve
                    docs = retrieve(result, collection, config)
                    # Evaluate
                    eval_result = evaluate(
                        result["transformer"],
                        query,
                        result["transformed_queries"],
                        docs,
                        config
                    )
                    all_results.append(eval_result)

                # Show transformed queries
                st.markdown("**🔄 Transformed Query:**")
                for q in result["transformed_queries"]:
                    st.info(q)

                # Show scores
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Relevance",
                    eval_result["relevance_score"]
                )
                col2.metric(
                    "Faithfulness",
                    eval_result["faithfulness_score"]
                )
                col3.metric(
                    "Completeness",
                    eval_result["completeness_score"]
                )
                col4.metric(
                    "Overall",
                    eval_result["overall_score"]
                )

                # Show retrieved chunks
                with st.expander(
                    f"📄 Retrieved Chunks ({len(docs)})"
                ):
                    for j, doc in enumerate(docs):
                        st.markdown(f"**Chunk {j+1}:**")
                        st.text(doc.page_content[:500])
                        st.divider()

                # Show answer
                st.markdown("**💬 Generated Answer:**")
                st.success(eval_result["answer"])

        # ── Leaderboard ────────────────────────────────────────────
        if len(all_results) > 1:
            st.divider()
            st.subheader("🏆 Leaderboard")

            df = pd.DataFrame(all_results)[[
                "transformer",
                "chunks_retrieved",
                "relevance_score",
                "faithfulness_score",
                "completeness_score",
                "overall_score"
            ]].sort_values("overall_score", ascending=False)

            df.insert(0, "Rank", range(1, len(df) + 1))
            df.columns = [
                "Rank", "Transformer", "Chunks",
                "Relevance", "Faithfulness",
                "Completeness", "Overall"
            ]

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # Download button
            csv = pd.DataFrame(all_results).to_csv(index=False)
            st.download_button(
                label="⬇️ Download Full Results CSV",
                data=csv,
                file_name="transformer_results.csv",
                mime="text/csv"
            )

            # Winner callout
            winner = df.iloc[0]["Transformer"]
            score = df.iloc[0]["Overall"]
            st.success(f"🥇 Best transformer: **{winner}** — Overall score: **{score}**")