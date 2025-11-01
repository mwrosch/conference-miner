# Conference Speaker Enricher

Turn a conference agenda (PDF or URL) into a structured list of speakers with affiliations, their ~10 most recent publications, and a short 1–2 sentence research-focus summary.

## Features
- PDF text extraction (`pdfplumber`) and webpage scraping (`requests` + `BeautifulSoup4`)
- Speaker/affiliation parsing (spaCy NER + heuristics)
- Author disambiguation (Semantic Scholar search) with fuzzy-match to affiliations
- Publications via Semantic Scholar Graph API (optional key) with Crossref fallback
- Streamlit UI: upload PDF or paste URL, export CSV/JSON

## One‑click deploy on Streamlit Community Cloud
1. Push this folder to **GitHub** (minimum files: `app.py`, `conference_speaker_enricher.py`, `requirements.txt`).
2. Go to https://share.streamlit.io (Streamlit Community Cloud) → **New app** → point to your repo/branch, set **Main file path** to `app.py`.
3. (Optional) In **Advanced settings**, add environment variable `S2_API_KEY` with your Semantic Scholar key.
4. Deploy.

## Local run
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # first run only
streamlit run app.py
```

## Environment
- **Optional:** `S2_API_KEY` to improve rate limits against the Semantic Scholar API.
- Respect site `robots.txt` and terms when scraping.

## FAQ
- **It found the wrong author.** Add more affiliation context in your agenda or edit the detected affiliation; the app uses fuzzy matching to pick the best author profile.
- **No speakers detected.** Agendas vary—try a cleaner source or edit the PDF; you can also tweak heuristics in `conference_speaker_enricher.py`.
- **Non-English agendas.** Swap the spaCy model and hints lists for your language.
