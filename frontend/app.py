"""Streamlit UI for SHL Assessment Recommendation Engine."""

from __future__ import annotations

import os
import requests
import streamlit as st

# ======================================================
# Configuration
# ======================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="🎯",
    layout="wide",
)

# ======================================================
# Header
# ======================================================

st.title("🎯 SHL Assessment Recommendation Engine")

st.markdown(
    """
AI-powered assessment recommendation system using **FAISS**, **Sentence Transformers**,
**FastAPI**, and **Ollama LLM**.

Enter a **job role**, **skills**, or **hiring requirement**
to receive the most relevant SHL assessments.
"""
)

st.divider()

# ======================================================
# Sidebar
# ======================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=10,
        value=10,
    )

    st.divider()

    st.subheader("Backend Status")

    st.success("✅ FastAPI")
    st.success("✅ FAISS")
    st.success("✅ Ollama")

    st.info("Model: qwen2.5:1.5b")

    st.divider()

    st.subheader("🔍 Filters")

    duration_filter = st.selectbox(
        "Maximum Duration",
        ["Any", 15, 30, 45, 60],
        index=0,
    )

    remote_filter = st.radio(
        "Remote Support",
        ["Any", "Yes", "No"],
        index=0,
    )

    adaptive_filter = st.radio(
        "Adaptive Support",
        ["Any", "Yes", "No"],
        index=0,
    )

    test_type_filter = st.multiselect(
        "Test Type",
        ["K", "S", "P", "C", "A", "B", "D", "E"],
        default=[],
    )

    st.divider()

    st.markdown(
        """
### About

This application recommends SHL assessments using:

- Semantic Search (FAISS)
- Sentence Transformers
- Ollama LLM
- FastAPI
- Streamlit
"""
    )

# ======================================================
# Search Section
# ======================================================

st.subheader("🔍 Search Assessment")

query = st.text_input(
    "Enter Job Role / Skills",
    placeholder="Example: Azure Data Engineer with Spark and SQL",
)

jd_url = st.text_input(
    "Job Description URL (Optional)",
    placeholder="https://company.com/job-description",
)

search = st.button(
    "🚀 Get Recommendations",
    use_container_width=True,
)

# ======================================================
# Search
# ======================================================

if search:

    if not query.strip():
        st.warning("Please enter a job role.")
        st.stop()

    with st.spinner("Searching SHL assessments..."):

        try:

            response = requests.post(
                f"{API_URL}/recommend",
                json={"query": query},
                timeout=120,
            )

            response.raise_for_status()

            recommendations = response.json().get(
                "recommended_assessments",
                []
            )

            # ==================================================
            # Apply Filters
            # ==================================================

            filtered = []

            for item in recommendations:

                # ---------------- Duration ----------------

                if duration_filter != "Any":

                    duration = item.get("duration")

                    if duration is not None:

                        try:
                            if float(duration) > float(duration_filter):
                                continue
                        except Exception:
                            pass

                # ---------------- Remote ----------------

                if remote_filter != "Any":

                    if item.get(
                        "remote_support",
                        "No"
                    ) != remote_filter:
                        continue

                # ---------------- Adaptive ----------------

                if adaptive_filter != "Any":

                    if item.get(
                        "adaptive_support",
                        "No"
                    ) != adaptive_filter:
                        continue

                # ---------------- Test Type ----------------

                if test_type_filter:

                    types = item.get(
                        "test_type",
                        []
                    )

                    if isinstance(types, str):
                        types = [types]

                    if not any(
                        t in types
                        for t in test_type_filter
                    ):
                        continue

                filtered.append(item)

            recommendations = filtered[:top_k]

            if not recommendations:
                st.warning(
                    "No assessments match the selected filters."
                )
                st.stop()

            st.success(
                f"Found {len(recommendations)} recommendation(s)"
            )

            st.divider()

            st.subheader("📋 Recommended Assessments")

            for i, item in enumerate(
                recommendations,
                start=1,
            ):

                with st.container(border=True):

                    st.markdown(
                        f"### {i}. [{item['name']}]({item['url']})"
                    )

                    st.write(
                        item.get(
                            "description",
                            "No description available.",
                        )
                    )

                    types = item.get(
                        "test_type",
                        []
                    )

                    if isinstance(types, str):
                        types = [types]

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Duration",
                            item.get("duration") or "N/A",
                        )

                        st.metric(
                            "Adaptive Support",
                            item.get(
                                "adaptive_support",
                                "No",
                            ),
                        )

                    with col2:

                        st.metric(
                            "Remote Support",
                            item.get(
                                "remote_support",
                                "No",
                            ),
                        )

                        st.metric(
                            "Test Type",
                            ", ".join(types),
                        )

                    st.link_button(
                        "🔗 Open Assessment",
                        item["url"],
                        use_container_width=True,
                    )

                    st.divider()

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to FastAPI backend.\n\n"
                "Run:\n\n"
                "python -m uvicorn api.app:app --port 8001"
            )

        except requests.exceptions.Timeout:

            st.error("⏱ Request timed out.")

        except Exception as e:

            st.error(f"Error: {e}")