#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conference Speaker Enricher
===========================
Given a conference agenda PDF or webpage URL, extract speaker names & affiliations,
fetch each speaker's ~10 most recent publications, and produce a short research-focus summary.

Features
--------
- Input: PDF path or webpage URL
- Extraction: pdfplumber for PDFs; requests+BeautifulSoup4 for webpages
- NER: spaCy (PERSON, ORG) with heuristics near "speaker", "keynote", etc.
- Author resolution: Semantic Scholar Graph API v1 (optional API key), fuzzy-scored with affiliation
- Publications: Semantic Scholar (preferred) or Crossref (fallback, open API)
- Summarization: keyword-scored phrases from titles/abstracts into 1-2 sentences
- Output: CSV and JSON

Install
-------
pip install pdfplumber beautifulsoup4 requests lxml spacy rapidfuzz python-dateutil tqdm
python -m spacy download en_core_web_sm

Environment (optional)
----------------------
- S2_API_KEY: Semantic Scholar API key (improves rate limits). Omit to run unauthenticated.

Usage
-----
python conference_speaker_enricher.py --pdf path/to/agenda.pdf --out out_dir
python conference_speaker_enricher.py --url https://conf.org/agenda --out out_dir
"""

import os
import re
import json
import csv
import time
import math
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
import pdfplumber

import spacy
from rapidfuzz import fuzz, process
from dateutil import parser as dateparser
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("S2_API_KEY", "").strip()
CROSSREF_BASE = "https://api.crossref.org"

HEADERS = {
    "User-Agent": "ConferenceSpeakerEnricher/1.0 (+https://example.com)"
}
if S2_API_KEY:
    HEADERS_S2 = {**HEADERS, "x-api-key": S2_API_KEY}
else:
    HEADERS_S2 = HEADERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ----------------------------
# Data Models
# ----------------------------
@dataclass
class Publication:
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None  # 'semantic_scholar' or 'crossref'

@dataclass
class Speaker:
    name: str
    affiliation: Optional[str] = None
    author_id: Optional[str] = None  # Semantic Scholar authorId if found
    publications: List[Publication] = None
    research_summary: Optional[str] = None

# ----------------------------
# Utilities
# ----------------------------
AFFIL_HINTS = [
    "university", "institute", "college", "school", "lab", "laboratory",
    "dept", "department", "centre", "center", "hospital", "clinic",
    "research", "inc", "ltd", "llc", "gmbh", "sa", "ag", "srl"
]

SPEAKER_HINTS = [
    "speaker", "keynote", "plenary", "panelist", "invited", "talk by",
    "presenter", "with:", "fireside chat", "in conversation with"
]

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def is_affiliation_like(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in AFFIL_HINTS)

def token_set_ratio(a: str, b: str) -> int:
    return fuzz.token_set_ratio((a or ""), (b or ""))

def top_keywords(texts: List[str], k: int = 12) -> List[str]:
    """Very simple keyword scorer (no stopword list to keep deps small)."""
    from collections import Counter
    words = []
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", t or ""):
            words.append(w.lower())
    # crude stopwords
    stops = set("""the and for with from this that have has are were was into over such into into
        about into within across among between beyond using used via via while into due into whose
        into using toward toward upon not into on at by of in a an to is be as it we they you he she 
        their our its than then however therefore moreover indeed into""".split())
    words = [w for w in words if w not in stops]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(k)]

def summarize_focus(publications: List[Publication]) -> str:
    titles = [p.title for p in publications if p.title]
    if not titles:
        return "Focus not clear from available records."
    kws = top_keywords(titles, k=10)
    # Short two-sentence template
    main = ", ".join(kws[:5]) if kws else "their field"
    also = ", ".join(kws[5:10]) if len(kws) > 5 else ""
    s1 = f"Their recent work centers on {main}."
    s2 = f"They also publish on {also}." if also else ""
    return normalize_space(f"{s1} {s2}")

# ----------------------------
# Extraction
# ----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)

def extract_text_from_url(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # Prefer main content if present
    main = soup.find("main") or soup.find("article") or soup.find("body")
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return text

# ----------------------------
# Speaker & affiliation parsing
# ----------------------------
def parse_speakers(text: str, nlp=None) -> List[Speaker]:
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    lines = [normalize_space(l) for l in text.splitlines() if normalize_space(l)]
    candidates: Dict[str, Dict[str, int]] = {}
    # Heuristic: look for PERSON entities; sniff nearby lines for affiliation-like org strings
    for i, line in enumerate(lines):
        low = line.lower()
        hinty = any(h in low for h in SPEAKER_HINTS) or bool(re.search(r"\b(Ph\.?D\.?|Prof\.?|Dr\.?)\b", line))
        if hinty or True:
            doc = nlp(line)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            orgs = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "GPE")]
            # Nearby lines may contain affiliation
            neighborhood = " ".join(lines[max(0, i-2): min(len(lines), i+3)])
            doc2 = nlp(neighborhood)
            orgs_near = [ent.text for ent in doc2.ents if ent.label_ in ("ORG", "GPE")]
            possible_affils = orgs + orgs_near
            for p in persons:
                if len(p.split()) < 2 or len(p) < 5:
                    continue
                key = normalize_space(p)
                affil = next((a for a in possible_affils if is_affiliation_like(a)), None)
                if key not in candidates:
                    candidates[key] = {"count": 0}
                candidates[key]["count"] += 1
                if affil:
                    candidates[key]["affil"] = affil
    # Convert to Speaker list, keep top occurrences
    items = sorted(candidates.items(), key=lambda kv: kv[1]["count"], reverse=True)
    speakers = []
    seen = set()
    for name, meta in items:
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        speakers.append(Speaker(name=name, affiliation=meta.get("affil")))
    return speakers

# ----------------------------
# Semantic Scholar integration
# ----------------------------
def s2_author_search(name: str) -> List[dict]:
    params = {"query": name, "limit": 5, "fields": "name,affiliations,url,homepage,aliases"}
    url = f"{S2_BASE}/author/search?{urlencode(params)}"
    r = requests.get(url, headers=HEADERS_S2, timeout=30)
    if r.status_code >= 400:
        logging.warning("Semantic Scholar search failed: %s %s", r.status_code, r.text[:200])
        return []
    data = r.json()
    return data.get("data", []) or data.get("authors", []) or []

def pick_best_author(candidates: List[dict], target_affil: Optional[str]) -> Optional[dict]:
    if not candidates:
        return None
    if not target_affil:
        # pick the first (S2 sorts by relevance)
        return candidates[0]
    best, best_score = None, -1
    for c in candidates:
        affs = " ".join(c.get("affiliations", []) or [])
        score = token_set_ratio(affs, target_affil)
        if score > best_score:
            best, best_score = c, score
    return best

def s2_recent_pubs(author_id: str, limit: int = 10) -> List[Publication]:
    params = {
        "limit": limit,
        "fields": "title,year,venue,url,externalIds,abstract"
    }
    url = f"{S2_BASE}/author/{author_id}/papers?{urlencode(params)}"
    r = requests.get(url, headers=HEADERS_S2, timeout=30)
    if r.status_code >= 400:
        logging.warning("Semantic Scholar papers failed: %s %s", r.status_code, r.text[:200])
        return []
    data = r.json()
    papers = data.get("data", []) or data.get("papers", []) or []
    pubs = []
    for p in papers:
        pubs.append(Publication(
            title=p.get("title"),
            year=p.get("year"),
            venue=p.get("venue"),
            url=p.get("url"),
            source="semantic_scholar",
        ))
    # sort by year desc, tie-breaker by title
    pubs.sort(key=lambda x: (x.year or 0, x.title or ""), reverse=True)
    return pubs[:limit]

# ----------------------------
# Crossref fallback
# ----------------------------
def crossref_recent_pubs(name: str, affiliation: Optional[str], limit: int = 10) -> List[Publication]:
    params = {
        "query.author": name,
        "rows": limit * 3,  # overfetch then filter
        "sort": "published", "order": "desc"
    }
    url = f"{CROSSREF_BASE}/works?{urlencode(params)}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code >= 400:
        logging.warning("Crossref failed: %s %s", r.status_code, r.text[:200])
        return []
    items = (r.json().get("message", {}) or {}).get("items", []) or []
    pubs = []
    for it in items:
        title = (it.get("title") or [""])[0]
        date_parts = (it.get("issued") or {}).get("date-parts", [[None]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        container = it.get("container-title") or []
        venue = container[0] if container else None
        url = it.get("URL")
        # Optional filter by affiliation token match in author list
        if affiliation:
            auths = it.get("author") or []
            aff_text = " ".join(a.get("affiliation", [{}])[0].get("name", "") for a in auths if a.get("affiliation"))
            if aff_text and token_set_ratio(aff_text, affiliation) < 40:
                continue
        pubs.append(Publication(title=title, year=year, venue=venue, url=url, source="crossref"))
    # sort
    pubs.sort(key=lambda x: (x.year or 0, x.title or ""), reverse=True)
    return pubs[:limit]

# ----------------------------
# Orchestrator
# ----------------------------
def enrich_speakers(speakers: List[Speaker], per_author_limit: int = 10, sleep: float = 0.8) -> List[Speaker]:
    out = []
    for spk in tqdm(speakers, desc="Enriching speakers"):
        # Find author
        candidates = s2_author_search(spk.name)
        best = pick_best_author(candidates, spk.affiliation)
        pubs: List[Publication] = []
        if best and best.get("authorId"):
            spk.author_id = str(best.get("authorId"))
            pubs = s2_recent_pubs(spk.author_id, per_author_limit)
        if not pubs:
            pubs = crossref_recent_pubs(spk.name, spk.affiliation, per_author_limit)
        spk.publications = pubs
        spk.research_summary = summarize_focus(pubs)
        out.append(spk)
        time.sleep(sleep)  # be polite to APIs
    return out

def speakers_to_csv_json(speakers: List[Speaker], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # JSON
    jpath = os.path.join(out_dir, "speakers_enriched.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in speakers], f, ensure_ascii=False, indent=2)
    # CSV (flatten top pubs titles)
    cpath = os.path.join(out_dir, "speakers_enriched.csv")
    cols = ["name", "affiliation", "author_id", "research_summary", "pub_titles"]
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in speakers:
            titles = "; ".join((p.title or "").strip() for p in (s.publications or [])[:10])
            w.writerow([s.name, s.affiliation or "", s.author_id or "", s.research_summary or "", titles])
    return jpath, cpath

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract speakers from PDF or URL and enrich with publications.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to conference agenda PDF")
    src.add_argument("--url", help="Conference agenda URL")
    ap.add_argument("--out", default="out", help="Output directory (default: out)")
    ap.add_argument("--limit", type=int, default=10, help="Publications per speaker (default: 10)")
    ap.add_argument("--max-speakers", type=int, default=100, help="Max speakers to process")
    args = ap.parse_args()

    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit("spaCy model not found. Run: python -m spacy download en_core_web_sm")

    if args.pdf:
        logging.info("Extracting text from PDF: %s", args.pdf)
        text = extract_text_from_pdf(args.pdf)
    else:
        logging.info("Extracting text from URL: %s", args.url)
        text = extract_text_from_url(args.url)

    logging.info("Parsing speakers...")
    speakers = parse_speakers(text, nlp=nlp)
    if not speakers:
        logging.warning("No speakers detected. You may need to adjust heuristics or provide a cleaner source.")
        speakers = []

    if args.max_speakers and len(speakers) > args.max_speakers:
        speakers = speakers[:args.max_speakers]

    logging.info("Enriching %d speakers...", len(speakers))
    enriched = enrich_speakers(speakers, per_author_limit=args.limit)
    jpath, cpath = speakers_to_csv_json(enriched, args.out)
    logging.info("Done. Wrote: %s and %s", jpath, cpath)

if __name__ == "__main__":
    main()
