import re
import time
import html
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

from src.explainer import analyze_article

RSS_FEEDS = {
    "🌍 Top Stories": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "🏛️ Politics": "https://news.google.com/rss/headlines/section/topic/POLITICS?hl=en-US&gl=US&ceid=US:en",
    "💻 Technology & AI": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "💼 Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    "🔬 Science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
    "🏥 Health": "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-US&gl=US&ceid=US:en",
    "🌐 World News": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
}

_FEED_CACHE = {}
_CACHE_EXPIRY = 300  # 5 minutes


def fetch_live_headlines(
    category_or_url: str = "🌍 Top Stories",
    search_query: Optional[str] = None,
    limit: int = 25,
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetches live news headlines from RSS feeds, parses the titles and publishers,
    and runs the AI fake news & reliability prediction pipeline on each item.
    """
    now = time.time()
    cache_key = f"{category_or_url}::{search_query}::{limit}"

    if not force_refresh and cache_key in _FEED_CACHE:
        cached_time, cached_data = _FEED_CACHE[cache_key]
        if now - cached_time < _CACHE_EXPIRY:
            return cached_data

    # Determine RSS URL
    if search_query and search_query.strip():
        encoded_q = urllib.parse.quote(search_query.strip())
        feed_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
    else:
        feed_url = RSS_FEEDS.get(category_or_url, RSS_FEEDS["🌍 Top Stories"])

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    raw_items = []
    try:
        req = urllib.request.Request(feed_url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            for item in items[:limit]:
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_elem = item.find("pubDate")
                source_elem = item.find("source")

                title_text = title_elem.text if title_elem is not None else ""
                link_text = link_elem.text if link_elem is not None else "#"
                pub_date = pub_elem.text if pub_elem is not None else ""
                source_name = source_elem.text if source_elem is not None else ""

                if not title_text:
                    continue

                # Google News RSS format is usually: "Headline text - Source Name"
                clean_title = title_text
                publisher = source_name
                if " - " in title_text:
                    parts = title_text.rsplit(" - ", 1)
                    clean_title = parts[0].strip()
                    if not publisher:
                        publisher = parts[1].strip()

                raw_items.append({
                    "title": html.unescape(clean_title),
                    "full_title": html.unescape(title_text),
                    "publisher": publisher or "External Feed",
                    "link": link_text,
                    "pub_date": pub_date
                })
    except Exception as e:
        print(f"Error fetching live RSS feed ({feed_url}): {e}")
        return []

    # Run predictions on all fetched headlines
    analyzed_items = []
    for item in raw_items:
        headline = item["title"]
        diag = analyze_article(headline)

        if "error" in diag:
            continue

        analyzed_items.append({
            "title": item["title"],
            "publisher": item["publisher"],
            "link": item["link"],
            "pub_date": item["pub_date"],
            "verdict": diag["verdict"],
            "confidence_score": diag["confidence_score"],
            "prob_reliable": diag["prob_reliable"],
            "prob_misleading": diag["prob_misleading"],
            "tier": diag["tier"],
            "verdict_color": diag["verdict_color"],
            "sensationalism_score": diag["linguistic_patterns"]["sensationalism_score"],
            "caps_ratio": diag["linguistic_patterns"]["caps_ratio"],
            "subjectivity": diag["linguistic_patterns"]["subjectivity"],
            "exclamation_count": diag["linguistic_patterns"]["exclamation_count"],
            "top_indicators": (
                diag["top_reliable_indicators"][:3]
                if diag["verdict"] == "Reliable"
                else diag["top_misleading_indicators"][:3]
            ),
            "reasoning": diag["reasoning"],
            "highlighted_html": diag["highlighted_html"]
        })

    _FEED_CACHE[cache_key] = (now, analyzed_items)
    return analyzed_items
