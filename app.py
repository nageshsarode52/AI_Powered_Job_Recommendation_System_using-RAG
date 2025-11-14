# app.py
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyBc1mwtkoB9PY-WJ93WP0KOcELrTadOM5U"

# app.pyset GOOGLE_API_KEY=AIzaSyBc1mwtkoB9PY-WJ93WP0KOcELrTadOM5U && python test_gemini.py
import os
import streamlit as st
import pandas as pd
from src.retriever import retrieve_top_k, compute_match_scores, load_store

# Optional: If you have Gemini key and want to enable later, set env var here.
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_KEY"

st.set_page_config(page_title="AI Job Recommender", layout="wide", initial_sidebar_state="collapsed")

# ---------- Styles ----------
PAGE_CSS = """
<style>
/* Page background and container */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
}

/* Card style for each job */
.job-card {
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
  box-shadow: 0 4px 14px rgba(17,24,39,0.06);
  background: #ffffff;
}

/* Title and small text */
.job-title {
  font-size: 18px;
  font-weight: 600;
  color: #0f172a;
}
.job-company {
  font-size: 14px;
  color: #475569;
  margin-bottom: 6px;
}

.header {
  background: linear-gradient(90deg, #0ea5e9, #7c3aed);
  color: white;
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(124,58,237,0.12);
}

.small-muted {
  color: #64748b;
  font-size: 13px;
}

/* Footer */
.footer {
  color: #475569;
  font-size: 13px;
  padding-top: 12px;
  padding-bottom: 24px;
  text-align: center;
}
</style>
"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ---------- Header ----------
with st.container():
    st.markdown('<div class="header">', unsafe_allow_html=True)
    col1, col2 = st.columns([8,2])
    with col1:
        st.markdown("## 💼 AI-Powered Job Recommendation System")
        st.markdown("<div class='small-muted'>Personalized job matches using semantic search (FAISS) and transformer embeddings. Enter your skills or upload a resume to begin.</div>", unsafe_allow_html=True)
    with col2:
        # optional image placeholder
        None
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")  # spacing

# ---------- Main two-column layout ----------
left_col, right_col = st.columns([3,7], gap="large")

with left_col:
    st.markdown("### 🔎 Search")
    with st.form("profile_form"):
        skills = st.text_area("Enter your skills / profile", placeholder="e.g. Python, SQL, Pandas, Power BI", height=120)
        preferred_location = st.text_input("Preferred location (optional)", placeholder="e.g. Pune, Bengaluru, Remote")
        job_type = st.selectbox("Job type", ["Any", "Full-time", "Internship", "Part-time", "Contract"])
        resume = st.file_uploader("Upload resume (optional, .pdf/.txt/.docx)", type=["pdf","txt","docx"])
        submitted = st.form_submit_button("Get Recommendations")
    st.markdown("---")
    st.markdown("### ⚙️ Options")
    st.caption("Results are ranked by semantic similarity. Use the location box to prefer jobs in a city.")
    st.markdown("---")
    st.markdown("### 📥 Download")
    st.markdown("Export the last search results as CSV (after running a search).")
    if "last_results" in st.session_state and st.session_state["last_results"]:
        df_download = pd.DataFrame(st.session_state["last_results"])
        csv_bytes = df_download.to_csv(index=False).encode("utf-8")
        st.download_button("Download last results as CSV", data=csv_bytes, file_name="job_recommendations.csv")
    else:
        st.info("Run a search to enable the download button.")

with right_col:
    # Load FAISS store once (this will also show error if missing)
    try:
        load_store()
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.stop()

    st.markdown("### 🔍 Results")
    if not submitted:
        st.info("Enter your skills and click *Get Recommendations* to see personalized job matches.")
    else:
        if not skills or not skills.strip():
            st.warning("Please enter some skills or profile text to search.")
        else:
            # Build profile text
            profile_text = f"Skills: {skills}. "
            if preferred_location:
                profile_text += f"Preferred location: {preferred_location}. "
            if job_type and job_type != "Any":
                profile_text += f"Job type: {job_type}."

            with st.spinner("Searching and ranking jobs..."):
                retrieved = retrieve_top_k(profile_text, k=12)
                ranked = compute_match_scores(retrieved)

            if not ranked:
                st.info("No matching jobs found. Try different skills or broader keywords.")
            else:
                # Save to session for download
                st.session_state["last_results"] = ranked

                # Use Gemini reasoning (if available) — safe call inside try/except
                try:
                    # import here to avoid hard dependency if LangChain/Gemini not installed
                    from src.retriever import explain_recommendations
                    with st.spinner("🤖 Gemini is analyzing your job matches..."):
                        reasoning_text = explain_recommendations(profile_text, ranked[:5])  # Top 5 jobs only
                    st.markdown("### 🤖 Gemini's Explanation For These Recommendations")
                    st.write(reasoning_text)
                except Exception as e:
                    # If Gemini not installed / key missing / any error, show a small notice but continue
                    st.info("Gemini reasoning not available (optional). To enable it, install the Gemini/langchain packages and set your API key.")
                    # Optionally log error in session or console for debugging:
                    # st.write(f"Gemini error: {e}")

                # Display summary row
                st.markdown(f"*Showing top {len(ranked)} results* — sorted by match score.")
                st.markdown("")

                # Render each job as a card
                for i, job in enumerate(ranked, start=1):
                    # Card container
                    st.markdown('<div class="job-card">', unsafe_allow_html=True)
                    # Title and company
                    st.markdown(f"<div class='job-title'>{i}. {job['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='job-company'>{job.get('company','')}</div>", unsafe_allow_html=True)
                    # meta row
                    cols = st.columns([3,1,1])
                    with cols[0]:
                        st.markdown(f"*📍 Location:* {job.get('location','N/A')}")
                    with cols[1]:
                        st.markdown(f"*💯 Score:* {job.get('match_score',0)}%")
                    with cols[2]:
                        # progress bar small (expects 0.0-1.0)
                        pct = job.get('match_score',0)
                        try:
                            st.progress(float(pct) / 100)
                        except:
                            st.progress(0)
                    # description
                    st.write(job.get('description','')[:500] + ("..." if len(job.get('description',''))>500 else ""))
                    # actions row
                    a1, a2 = st.columns([1,4])
                    with a1:
                        if job.get('url'):
                            st.markdown(f"[🔗 Link]({job.get('url')})")
                        else:
                            st.markdown(" ")
                    with a2:
                        st.markdown("<div style='text-align:right; color:#64748b; font-size:13px;'>Retrieved by semantic similarity</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.success("Tip: refine skills or location to narrow results.")

# ---------- Footer ----------
st.markdown("<div class='footer'>Made with ❤️ — AI Job Recommender • Powered by FAISS + SentenceTransformers • (Gemini integration optional)</div>", unsafe_allow_html=True