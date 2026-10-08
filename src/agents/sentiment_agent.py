# src/agents/sentiment_agent.py
#
# Phase 4c: FinBERT sentiment agent. Fetches recent BTC headlines
# (CryptoCompare News API, free/no-auth), scores with FinBERT, aggregates
# to daily sentiment_score. Fails soft (skips feature, no synthetic
# default) — same contract as fear_greed_agent.py / onchain_agent.py.

import requests
import pandas as pd
import time

NEWS_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_pipeline = None


def _get_pipeline():
    """Lazy-load so importing this module doesn't force transformers/torch
    load for scripts that don't need sentiment."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    return _pipeline

def fetch_headlines(query="bitcoin", maxrecords=100, max_retries=4):
    """GDELT DOC 2.0 API — free, no auth, no signup. Returns recent
    articles (typically last few days of coverage). GDELT rate-limits
    requests without a User-Agent and on rapid retries."""
    headers = {"User-Agent": "Mozilla/5.0 (research script; AI_Trading_Suite)"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                NEWS_URL,
                params={
                    "query": f"{query} sourcelang:english",
                    "mode": "ArtList",
                    "maxrecords": maxrecords,
                    "format": "json",
                    "sort": "DateDesc",
                },
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[sentiment_agent] 429 rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            if not articles:
                raise RuntimeError("NO articles")
            df = pd.DataFrame(articles)[["seendate", "title"]]
            df["date"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ").dt.date
            return df[["date", "title"]]
        except requests.exceptions.Timeout:
            wait = 5 * (attempt + 1)
            print(f"[sentiment_agent] timed out, retrying in {wait}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            raise RuntimeError(f"[sentiment_agent] fetch failed: {e}")
    raise RuntimeError("[sentiment_agent] gave up after retries.")

def score_headlines(headlines_df):
    """Per-day mean(P(positive) - P(negative)), range [-1, 1]."""
    try:
        clf = _get_pipeline()
        results = clf(headlines_df["title"].tolist(), truncation=True)
        signed = []
        for r in results:
            label, conf = r["label"].lower(), r["score"]
            signed.append(conf if label == "positive" else (-conf if label == "negative" else 0.0))
        out = headlines_df.copy()
        out["score"] = signed
        return out.groupby("date")["score"].mean().reset_index(name="sentiment_score")
    except Exception as e:
        raise RuntimeError(f"[sentiment_agent] scoring failed: {e}")


def merge_sentiment(df):
    """Merges daily sentiment_score into hourly OHLCV df by date.
    If in any process any problem came it will stop all the process and raise RuntimeError."""
    headlines = fetch_headlines()

    daily = score_headlines(headlines)

    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df = df.merge(daily, on="date", how="left")
    df.drop(columns=["date"], inplace=True)
    return df