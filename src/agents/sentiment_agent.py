# src/agents/sentiment_agent.py
"""
Alternative data extractor for GDELT news sentiment and FinBERT inference.
Fetches recent Bitcoin news headlines via the public GDELT DOC 2.0 API, scores them
with ProsusAI FinBERT transformer pipeline, aggregates daily signed sentiment scores,
and merges them across hourly candles.
"""

from typing import Any, Optional
import time
import pandas as pd
import requests

NEWS_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_pipeline: Optional[Any] = None


def _get_pipeline() -> Any:
    """
    Lazy-load FinBERT pipeline so importing this module does not trigger
    heavy PyTorch/transformers memory allocation unless explicitly invoked.
    """
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    return _pipeline


def fetch_headlines(
    query: str = "bitcoin",
    maxrecords: int = 100,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Fetches recent news articles from GDELT DOC 2.0 API (free, unauthenticated).

    Args:
        query (str): Keyword query for article filtering.
        maxrecords (int): Maximum number of headlines to retrieve.
        max_retries (int): Number of retry attempts on rate limits or network issues.

    Returns:
        pd.DataFrame: DataFrame containing 'date' and 'title' columns.

    Raises:
        RuntimeError: If headline acquisition fails after all retry attempts.
    """
    headers = {"User-Agent": "Mozilla/5.0 (research script; Mudra_Quant_Platform)"}
    last_error: Optional[Exception] = None

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
                raise ValueError("GDELT returned empty articles list.")

            df = pd.DataFrame(articles)
            if "seendate" not in df.columns or "title" not in df.columns:
                raise KeyError("GDELT payload missing 'seendate' or 'title' fields.")

            df["date"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce").dt.date
            df = df.dropna(subset=["date", "title"])
            return df[["date", "title"]].reset_index(drop=True)

        except requests.exceptions.Timeout:
            wait = 5 * (attempt + 1)
            print(f"[sentiment_agent] Request timed out, retrying in {wait}s...")
            time.sleep(wait)
            continue
        except (requests.RequestException, ValueError, KeyError) as e:
            last_error = e
            wait = 3 * (attempt + 1)
            print(f"[sentiment_agent] Transient error on attempt {attempt + 1}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            continue

    raise RuntimeError(f"[sentiment_agent] Headline acquisition failed after {max_retries} retries: {last_error}")


def score_headlines(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs FinBERT sentiment classification on article titles and aggregates
    daily mean signed sentiment: P(positive) - P(negative) in [-1.0, 1.0].

    Args:
        headlines_df (pd.DataFrame): DataFrame containing 'date' and 'title'.

    Returns:
        pd.DataFrame: Aggregated DataFrame with 'date' and 'sentiment_score'.

    Raises:
        RuntimeError: If model inference fails.
    """
    if headlines_df.empty:
        return pd.DataFrame(columns=["date", "sentiment_score"])

    try:
        clf = _get_pipeline()
        results = clf(headlines_df["title"].tolist(), truncation=True)
        signed = []
        for r in results:
            label, conf = r["label"].lower(), float(r["score"])
            signed.append(conf if label == "positive" else (-conf if label == "negative" else 0.0))

        out = headlines_df.copy()
        out["score"] = signed
        return out.groupby("date")["score"].mean().reset_index(name="sentiment_score")

    except Exception as e:
        raise RuntimeError(f"[sentiment_agent] FinBERT scoring failed: {e}")


def merge_sentiment(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Merges daily sentiment scores into an hourly OHLCV DataFrame by date.
    Upholds strict data integrity: raises RuntimeError if headline ETL fails.

    Args:
        df (pd.DataFrame): Candlestick DataFrame.
        timestamp_col (str): Timestamp column name.

    Returns:
        pd.DataFrame: DataFrame with merged 'sentiment_score' column.
    """
    if df.empty:
        return df.copy()

    headlines = fetch_headlines()
    daily = score_headlines(headlines)

    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date

    # Prevent duplicate column suffixes (_x, _y) on repeated runs
    if "sentiment_score" in df.columns:
        df = df.drop(columns=["sentiment_score"])

    df = df.merge(daily, left_on="_date", right_on="date", how="left")
    return df.drop(columns=["_date", "date"], errors="ignore")


if __name__ == "__main__":
    print("[INFO] Testing headline fetch from GDELT...")
    try:
        sample_headlines = fetch_headlines(maxrecords=5)
        print(f"[SUCCESS] Fetched {len(sample_headlines)} headlines:\n", sample_headlines)
    except Exception as err:
        print(f"[ERROR] Headline fetch failed: {err}")