import io
import os
import json
import time
import pandas as pd
import streamlit as st

# Local module (same directory)
import conference_speaker_enricher as cse

st.set_page_config(page_title="Conference Speaker Enricher", page_icon="🧑‍🎓", layout="wide")

st.title("🧑‍🎓 Conference Speaker Enricher")
st.caption("Upload a PDF agenda or paste a conference URL to extract speakers, fetch recent publications, and summarize research focus.")

with st.expander("Settings", expanded=False):
    limit = st.number_input("Publications per speaker", min_value=3, max_value=25, value=10, step=1)
    max_speakers = st.number_input("Max speakers to process", min_value=1, max_value=500, value=100, step=1)
    sleep = st.number_input("Delay between authors (seconds)", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
    s2_key = st.text_input("Semantic Scholar API key (optional)", type="password")
    if s2_key:
        os.environ["S2_API_KEY"] = s2_key

tab_pdf, tab_url = st.tabs(["📄 PDF Agenda", "🌐 Website URL"])

text_source = None

with tab_pdf:
    uploaded_pdf = st.file_uploader("Upload Conference Agenda PDF", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Extract from PDF", type="primary", use_container_width=True):
            with st.spinner("Extracting text from PDF..."):
                # Save to a temp path for pdfplumber
                tmp_path = "uploaded_agenda.pdf"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                text_source = cse.extract_text_from_pdf(tmp_path)

with tab_url:
    url = st.text_input("Paste conference agenda URL", placeholder="https://example.org/agenda")
    if st.button("Extract from URL", type="primary", use_container_width=True, key="urlbtn"):
        if not url.strip():
            st.error("Please provide a URL.")
        else:
            with st.spinner("Fetching & parsing webpage..."):
                try:
                    text_source = cse.extract_text_from_url(url.strip())
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")

if text_source:
    st.success("Text source loaded. Parsing speakers...")
    # Load spaCy model with on-demand download fallback
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        with st.spinner("Downloading spaCy model (en_core_web_sm)..."):
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
        import spacy
        nlp = spacy.load("en_core_web_sm")

    speakers = cse.parse_speakers(text_source, nlp=nlp)
    if not speakers:
        st.warning("No speakers detected. Try a different source or tweak the agenda content.")
    else:
        st.info(f"Detected {len(speakers)} candidate speakers. Processing up to {max_speakers}.")
        speakers = speakers[:max_speakers]

        progress = st.progress(0)
        enriched_all = []
        for i, sp in enumerate(speakers, start=1):
            enriched = cse.enrich_speakers([sp], per_author_limit=int(limit), sleep=float(sleep))
            enriched_all.extend(enriched)
            progress.progress(i / len(speakers))
        progress.empty()

        # Build a DataFrame
        rows = []
        for s in enriched_all:
            titles = [p.title for p in (s.publications or [])]
            venues = [p.venue for p in (s.publications or [])]
            years = [p.year for p in (s.publications or [])]
            rows.append({
                "name": s.name,
                "affiliation": s.affiliation or "",
                "author_id": s.author_id or "",
                "research_summary": s.research_summary or "",
                "pub_titles": titles,
                "pub_years": years,
                "pub_venues": venues,
            })
        df = pd.DataFrame(rows)

        st.subheader("Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Downloads
        json_bytes = io.BytesIO(json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"))
        csv_df = df.copy()
        # Flatten list columns for CSV readability
        csv_df["pub_titles"] = csv_df["pub_titles"].apply(lambda x: "; ".join(x))
        csv_df["pub_years"] = csv_df["pub_years"].apply(lambda x: "; ".join(str(y) for y in x if y is not None))
        csv_df["pub_venues"] = csv_df["pub_venues"].apply(lambda x: "; ".join(v for v in x if v))
        csv_bytes = io.BytesIO(csv_df.to_csv(index=False).encode("utf-8"))

        st.download_button("⬇️ Download JSON", data=json_bytes, file_name="speakers_enriched.json", mime="application/json")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="speakers_enriched.csv", mime="text/csv")

        st.caption("Tip: Set a Semantic Scholar API key in Settings for better reliability and rate limits.")
else:
    st.info("Upload a PDF or paste a URL to get started.")
