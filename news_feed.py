"""
News fetching and AI-powered sentiment analysis for the Trading Simulator.
Uses Google News RSS (free, no key needed) and Groq/Gemini for sentiment scoring.
"""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
import streamlit as st


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _get_key(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def _parse_google_news_rss(xml_text: str, limit: int) -> list[dict]:
    """Parse Google News RSS XML into article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        items = root.findall(".//item")
        for item in items[:limit]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            # Extract source from title (Google News format: "Title - Source")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]
            # Convert RFC 2822 date to ISO format
            iso_date = ""
            if pub_date:
                try:
                    dt = parsedate_to_datetime(pub_date)
                    iso_date = dt.isoformat()
                except Exception:
                    iso_date = pub_date
            articles.append({
                "title": title,
                "url": link,
                "publishedDate": iso_date,
                "source": source,
                "image": "",
                "snippet": "",
            })
    except Exception:
        pass
    return articles


@st.cache_data(ttl=300)
def get_stock_news(ticker: str, limit: int = 8) -> list[dict]:
    """Fetch news articles for a specific ticker via Google News RSS."""
    try:
        resp = requests.get(
            GOOGLE_NEWS_RSS,
            params={
                "q": f"{ticker} stock",
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return _parse_google_news_rss(resp.text, limit)
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_general_news(limit: int = 5) -> list[dict]:
    """Fetch general stock market news via Google News RSS."""
    try:
        resp = requests.get(
            GOOGLE_NEWS_RSS,
            params={
                "q": "stock market today",
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return _parse_google_news_rss(resp.text, limit)
    except Exception:
        return []


def _build_sentiment_prompt(headlines: list[str]) -> str:
    """Build a prompt to classify headlines as Bullish/Bearish/Neutral."""
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return f"""Classify each headline's stock market sentiment as Bullish, Bearish, or Neutral.
Return ONLY a JSON array with no extra text. Each element must have:
- "sentiment": "Bullish" or "Bearish" or "Neutral"
- "confidence": a float from 0.0 to 1.0

Example output for 2 headlines:
[{{"sentiment":"Bullish","confidence":0.8}},{{"sentiment":"Neutral","confidence":0.5}}]

Headlines:
{numbered}"""


@st.cache_data(ttl=300)
def analyze_sentiment_batch(headlines: tuple) -> list[dict]:
    """Score headlines using Groq (primary) or Gemini (fallback).

    Pass headlines as a tuple for Streamlit cache compatibility.
    """
    headlines = list(headlines)
    default = [{"sentiment": "Neutral", "confidence": 0.0} for _ in headlines]
    if not headlines:
        return default

    prompt = _build_sentiment_prompt(headlines)

    # Try Groq first
    groq_key = _get_key("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq

            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if len(parsed) == len(headlines):
                    return parsed
        except Exception:
            pass

    # Fallback to Gemini
    gemini_key = _get_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if len(parsed) == len(headlines):
                    return parsed
        except Exception:
            pass

    return default


def get_sentiment_color(sentiment: str, colors: dict) -> str:
    """Map sentiment to theme color."""
    mapping = {
        "Bullish": colors.get("GREEN", "#3fb950"),
        "Bearish": colors.get("RED", "#f85149"),
        "Neutral": colors.get("YELLOW", "#d29922"),
    }
    return mapping.get(sentiment, colors.get("YELLOW", "#d29922"))


def format_time_ago(published_date: str) -> str:
    """Convert ISO date string to a human-readable relative time."""
    if not published_date:
        return ""
    try:
        dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 0:
            return "just now"
        elif seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        else:
            return f"{seconds // 86400}d ago"
    except Exception:
        return ""
