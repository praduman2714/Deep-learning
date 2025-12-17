#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Malaria News Scraper (SearxNG – Page 1 only)

Output:
- Proper JSON (not JSONL)
- Full article schema:
  query, title, full_content, published_date, author, etc.
- Tracks scraping failures + reasons
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import re
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

import requests
import dateparser
from dateparser.search import search_dates
from bs4 import BeautifulSoup

# ---------------- NLP ----------------
import spacy
NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
NLP.enable_pipe("ner")

# ---------------- Summarization ----------------
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    HAS_TEXTRANK = True
except Exception:
    HAS_TEXTRANK = False

# ---------------- Extraction ----------------
try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from readability import Document
except Exception:
    Document = None

# ---------------- Constants ----------------
QUERY = "malaria"
BASE_URL = "https://searxng.merai.app"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
TIMEOUT = 25
MAX_BYTES = 2_000_000
TODAY = datetime.utcnow().date()

# ---------------- Stats ----------------
SCRAPE_STATS = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "failure_reasons": {}
}
FAILED_URLS = []
# =========================================================
# Helpers
# =========================================================
def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def http_get(url: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
            verify=False
        )
        if r.status_code != 200:
            return None
        return r.content[:MAX_BYTES].decode(r.encoding or "utf-8", errors="ignore")
    except Exception:
        return None


def extract_main_text(html: str):
    title, content = "", ""

    if trafilatura:
        try:
            content = trafilatura.extract(html, output="txt") or ""
        except Exception:
            pass

    if not content and Document:
        try:
            doc = Document(html)
            title = doc.short_title()
            soup = BeautifulSoup(doc.summary(), "lxml")
            content = soup.get_text(" ", strip=True)
        except Exception:
            pass

    if not content:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string if soup.title else ""
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        content = soup.get_text(" ", strip=True)

    return normalize_ws(title), normalize_ws(content)


def extract_author(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")

    meta_names = [
        "author",
        "article:author",
        "parsely-author"
    ]

    for m in soup.find_all("meta"):
        name = m.get("name") or m.get("property")
        if name and name.lower() in meta_names:
            return m.get("content")

    return None


def summarize(text: str) -> str:
    if not text:
        return ""
    if HAS_TEXTRANK:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            return " ".join(str(s) for s in summarizer(parser.document, 5))
        except Exception:
            pass
    return text[:1200]


def extract_entities(text: str) -> Dict[str, List[str]]:
    doc = NLP(text[:500_000])
    buckets = {}
    for ent in doc.ents:
        buckets.setdefault(ent.label_, set()).add(ent.text)
    return {k: sorted(v) for k, v in buckets.items()}


def extract_dates(text: str) -> List[str]:
    found = search_dates(text, add_detected_language=False)
    dates = set()
    if found:
        for _, dt in found:
            if dt.date() <= TODAY:
                dates.add(dt.date().isoformat())
    return sorted(dates, reverse=True)


# =========================================================
# SearxNG – GET, page 1 only
# =========================================================
def searxng_results():
    params = {
        "q": QUERY,
        "categories": "news",
        "language": "en",
        "format": "json",
        "pageno": 1,
        "safesearch": 1
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Referer": "https://searxng.merai.app"
    }

    r = requests.get(BASE_URL, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    for r in data.get("results", []):
        yield {
            "search_query": QUERY,
            "title": r.get("title"),
            "url": r.get("url"),
            "source": r.get("hostname"),
            "published_raw": r.get("publishedDate") or r.get("pubdate")
        }


# =========================================================
# Enrichment
# =========================================================
def enrich(item: Dict) -> Dict:
    SCRAPE_STATS["total"] += 1
    url = item["url"]
    uid = hashlib.md5(url.encode()).hexdigest()

    html = http_get(url)
    if not html:
        SCRAPE_STATS["failed"] += 1
        SCRAPE_STATS["failure_reasons"]["http_fetch_failed"] = (
            SCRAPE_STATS["failure_reasons"].get("http_fetch_failed", 0) + 1
        )

        FAILED_URLS.append({
            "url": url,
            "reason": "http_fetch_failed"
        })

        return {
            **item,
            "id": uid,
            "fetch_ok": False,
            "failure_reason": "http_fetch_failed"
        }

    title, content = extract_main_text(html)
    if not content:
        SCRAPE_STATS["failed"] += 1
        SCRAPE_STATS["failure_reasons"]["content_extraction_failed"] = (
            SCRAPE_STATS["failure_reasons"].get("content_extraction_failed", 0) + 1
        )
        return {
            **item,
            "id": uid,
            "fetch_ok": False,
            "failure_reason": "content_extraction_failed"
        }

    SCRAPE_STATS["success"] += 1

    timeline = extract_dates(content)
    published = (
        dateparser.parse(item.get("published_raw")).date().isoformat()
        if item.get("published_raw")
        else (timeline[0] if timeline else None)
    )

    return {
        "id": uid,
        "search_query": item["search_query"],
        "title": title or item["title"],
        "url": url,
        "source": item["source"],
        "author": extract_author(html),
        "published_date": published,
        "full_content": content,
        "summary": summarize(content),
        "entities": extract_entities(content),
        "timeline_dates": timeline,
        "fetch_ok": True
    }


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs("out", exist_ok=True)
    out_path = "out/news_malaria.json"

    articles = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(enrich, item) for item in searxng_results()]
        for f in as_completed(futures):
            articles.append(f.result())

    output = {
        "meta": {
            "query": QUERY,
            "total_attempted": SCRAPE_STATS["total"],
            "successful": SCRAPE_STATS["success"],
            "failed": SCRAPE_STATS["failed"],
            "failure_reasons": SCRAPE_STATS["failure_reasons"],
            "failed_urls_count": len(FAILED_URLS) 
        },
        "failed_urls": FAILED_URLS,
        "articles": sorted(
            articles,
            key=lambda x: x.get("published_date") or "",
            reverse=True
        )
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✅ Finished")
    print(json.dumps(output["meta"], indent=2))


if __name__ == "__main__":
    main()
