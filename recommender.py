"""
recommender.py
Core Machine Learning Recommendation Engine for MovieMind.
Implements Content-Based Filtering using TF-IDF and Cosine Similarity.
"""

import os
import ast
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def _safe_literal_eval(val: Any) -> Any:
    """Safely parse JSON-like string, trying fast json.loads first, then ast.literal_eval."""
    if not isinstance(val, str):
        return []
    val = val.strip()
    if not val or val == "[]" or val == "{}":
        return []
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return []


def _extract_names(val: Any) -> List[str]:
    """Extract list of 'name' values from parsed JSON-like list of dicts."""
    items = _safe_literal_eval(val)
    if isinstance(items, list):
        return [item.get("name", "").strip() for item in items if isinstance(item, dict) and item.get("name")]
    return []


def _extract_director(val: Any) -> str:
    """Fast extraction of Director name from crew JSON string."""
    if not isinstance(val, str) or "Director" not in val:
        return ""
    items = _safe_literal_eval(val)
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("job") == "Director":
                return item.get("name", "").strip()
    return ""


def _extract_top_cast(val: Any, top_n: int = 4) -> List[str]:
    """Extract top N prominent cast members from cast JSON-like list of dicts."""
    items = _safe_literal_eval(val)
    if isinstance(items, list):
        names = []
        for item in items[:top_n]:
            if isinstance(item, dict) and item.get("name"):
                names.append(item.get("name").strip())
        return names
    return []


def _collapse_token(name: str) -> str:
    """Collapse spaces and punctuation to create a distinct entity token."""
    return "".join(name.lower().split())


class MovieRecommender:
    """
    Content-Based Movie Recommender System.
    Loads TMDB 5000 movies & credits datasets, extracts rich metadata features,
    builds TF-IDF representations, and calculates cosine similarity rankings.
    """

    def __init__(
        self,
        movies_path: str = "data/tmdb_5000_movies.csv",
        credits_path: str = "data/tmdb_5000_credits.csv",
        posters_cache_path: str = "data/posters_cache.json"
    ):
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.posters_cache_path = posters_cache_path

        self.df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.title_to_idx: Dict[str, int] = {}
        self.id_to_idx: Dict[int, int] = {}
        self.all_genres: List[str] = []
        self.posters_cache: Dict[int, str] = {}

        self._load_and_train()

    def _load_posters_cache(self):
        """Load pre-cached poster paths if available."""
        if os.path.exists(self.posters_cache_path):
            try:
                with open(self.posters_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.posters_cache = {}
                    for k, v in data.items():
                        if v:
                            self.posters_cache[str(k)] = v
                            try:
                                self.posters_cache[int(k)] = v
                            except Exception:
                                pass
                logger.info(f"Loaded {len(data)} cached poster paths.")
            except Exception as e:
                logger.warning(f"Failed to load posters cache: {e}")

    def save_posters_cache(self):
        """Save poster paths cache to disk."""
        try:
            with open(self.posters_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.posters_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save posters cache: {e}")

    def _load_and_train(self):
        """Load datasets, clean, extract features, and construct the TF-IDF matrix."""
        logger.info("Loading TMDB 5000 datasets...")
        if not os.path.exists(self.movies_path):
            raise FileNotFoundError(f"Movies dataset not found at {self.movies_path}")
        if not os.path.exists(self.credits_path):
            raise FileNotFoundError(f"Credits dataset not found at {self.credits_path}")

        movies_df = pd.read_csv(self.movies_path)
        credits_df = pd.read_csv(self.credits_path)

        # Merge on movie id
        df = movies_df.merge(credits_df, left_on="id", right_on="movie_id", suffixes=("", "_credits"))

        # Drop duplicates by movie id or title
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

        # Basic cleaning
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        df["overview"] = df["overview"].fillna("").astype(str).str.strip()
        df["tagline"] = df["tagline"].fillna("").astype(str).str.strip()
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0.0)

        # Release year extraction
        df["release_date"] = df["release_date"].fillna("").astype(str)
        df["release_year"] = df["release_date"].apply(
            lambda d: d.split("-")[0] if len(d) >= 4 and d.split("-")[0].isdigit() else "N/A"
        )

        # Feature Extraction
        df["genres_list"] = df["genres"].apply(_extract_names)
        df["keywords_list"] = df["keywords"].apply(_extract_names)
        df["director"] = df["crew"].apply(_extract_director)
        df["cast_list"] = df["cast"].apply(lambda c: _extract_top_cast(c, top_n=4))

        # Formatted string representations for UI
        df["genres_display"] = df["genres_list"].apply(lambda g: ", ".join(g) if g else "General")
        df["cast_display"] = df["cast_list"].apply(lambda c: ", ".join(c) if c else "Not Listed")
        df["director_display"] = df["director"].apply(lambda d: d if d else "Unknown")

        # Load posters cache
        self._load_posters_cache()

        # Check if dataset has poster_path column already
        has_poster_col = "poster_path" in df.columns
        df["poster_path"] = df["id"].apply(
            lambda mid: df.loc[df["id"] == mid, "poster_path"].values[0]
            if has_poster_col and pd.notnull(df.loc[df["id"] == mid, "poster_path"].values[0])
            else self.posters_cache.get(mid, "")
        )

        # Extract unique genres
        genres_set = set()
        for g_list in df["genres_list"]:
            genres_set.update(g_list)
        self.all_genres = sorted(list(genres_set))

        # Construct combined feature soup
        logger.info("Engineering content features...")
        df["soup"] = df.apply(self._create_soup, axis=1)

        # Fit TF-IDF Vectorizer
        logger.info("Computing TF-IDF representation (unigrams + bigrams)...")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df["soup"])
        logger.info(f"TF-IDF matrix built with shape {self.tfidf_matrix.shape}")

        self.df = df

        # Quick lookup dictionaries
        self.title_to_idx = {title.lower(): idx for idx, title in enumerate(df["title"])}
        self.id_to_idx = {int(mid): idx for idx, mid in enumerate(df["id"])}

    @staticmethod
    def _create_soup(row: pd.Series) -> str:
        """
        Creates an enriched text representation for a movie.
        Emphasizes genres and keywords with 2x weighting and collapses entity names
        (director, actors) into single tokens to prevent accidental cross-matching.
        """
        overview = row["overview"]
        tagline = row["tagline"]

        # Genres: both standard words and collapsed tokens, repeated for higher weight
        genres_words = " ".join([g.lower() for g in row["genres_list"]])
        genres_tokens = " ".join([_collapse_token(g) for g in row["genres_list"]])
        genres_part = f"{genres_words} {genres_tokens} {genres_tokens}"

        # Keywords: standard and collapsed, repeated for higher weight
        keywords_words = " ".join([k.lower() for k in row["keywords_list"]])
        keywords_tokens = " ".join([_collapse_token(k) for k in row["keywords_list"]])
        keywords_part = f"{keywords_words} {keywords_tokens} {keywords_tokens}"

        # Top cast: collapsed tokens so 'Tom Cruise' doesn't match 'Tom Hanks'
        cast_tokens = " ".join([_collapse_token(c) for c in row["cast_list"]])

        # Director: collapsed token
        director_token = _collapse_token(row["director"]) if row["director"] else ""

        # Combine all parts
        soup = f"{overview} {tagline} {genres_part} {keywords_part} {cast_tokens} {director_token}"
        return soup

    def get_movie_by_title(self, title: str) -> Optional[pd.Series]:
        """Lookup a movie record by title (case-insensitive)."""
        idx = self.title_to_idx.get(title.strip().lower())
        if idx is not None:
            return self.df.iloc[idx]
        return None

    def get_movie_by_id(self, movie_id: int) -> Optional[pd.Series]:
        """Lookup a movie record by TMDB movie ID."""
        idx = self.id_to_idx.get(int(movie_id))
        if idx is not None:
            return self.df.iloc[idx]
        return None

    def search_titles(self, query: str, limit: int = 15) -> List[str]:
        """
        Search movie titles matching query.
        Returns titles ordered by prefix match first, then substring match, sorted by popularity.
        """
        query_clean = query.strip().lower()
        if not query_clean:
            # Default to top popular movies
            return self.df.sort_values(by="popularity", ascending=False)["title"].head(limit).tolist()

        matches = self.df[self.df["title"].str.lower().str.contains(query_clean, regex=False, na=False)].copy()
        if matches.empty:
            return []

        # Boost prefix matches
        matches["prefix_match"] = matches["title"].str.lower().str.startswith(query_clean)
        matches = matches.sort_values(by=["prefix_match", "popularity"], ascending=[False, False])
        return matches["title"].head(limit).tolist()

    def get_recommendations(
        self,
        movie_title_or_id: Any,
        top_k: int = 10,
        genre_filter: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate ML content-based recommendations for a given movie title or ID.

        Steps:
        1. Locate selected movie's index.
        2. Compute cosine similarity against all movies using linear_kernel.
        3. Exclude the selected movie itself.
        4. Apply genre filter if specified.
        5. Return top K ranked recommendations with similarity scores and metadata.
        """
        if isinstance(movie_title_or_id, int) or (isinstance(movie_title_or_id, str) and movie_title_or_id.isdigit()):
            idx = self.id_to_idx.get(int(movie_title_or_id))
        else:
            idx = self.title_to_idx.get(str(movie_title_or_id).strip().lower())

        if idx is None:
            logger.warning(f"Movie '{movie_title_or_id}' not found in catalogue.")
            return []

        # Cosine similarity between selected movie's TF-IDF vector and all movie vectors
        # Note: Since TF-IDF vectors from TfidfVectorizer are L2-normalized, linear_kernel computes cosine similarity!
        sim_scores = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Sort indices by similarity descending
        sorted_indices = sim_scores.argsort()[::-1]

        recommendations = []
        for i in sorted_indices:
            # Exclude selected movie itself
            if i == idx:
                continue

            row = self.df.iloc[i]

            # Apply genre filter if provided
            if genre_filter and genre_filter != "All Genres":
                if genre_filter not in row["genres_list"]:
                    continue

            score = float(sim_scores[i])
            # Ensure similarity is between 0 and 1
            score = max(0.0, min(1.0, score))

            mid = int(row["id"])
            s_mid = str(mid)
            poster_url = None

            # Check df row poster_path
            poster_path = row["poster_path"]
            # Check posters_cache
            if not poster_path or not isinstance(poster_path, str) or not poster_path.strip():
                poster_path = self.posters_cache.get(mid) or self.posters_cache.get(s_mid)

            if poster_path and isinstance(poster_path, str) and poster_path.strip():
                if poster_path.startswith("http"):
                    poster_url = poster_path
                else:
                    poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path.strip()}"
            else:
                poster_url = self.fetch_poster_for_movie(mid, api_key=api_key)

            recommendations.append({
                "id": int(row["id"]),
                "title": row["title"],
                "release_year": row["release_year"],
                "genres": row["genres_list"],
                "genres_display": row["genres_display"],
                "vote_average": round(float(row["vote_average"]), 1),
                "vote_count": int(row["vote_count"]),
                "popularity": round(float(row["popularity"]), 1),
                "runtime": int(row["runtime"]) if row["runtime"] else "N/A",
                "director": row["director_display"],
                "cast": row["cast_display"],
                "overview": row["overview"],
                "tagline": row["tagline"],
                "similarity_score": score,
                "similarity_percentage": f"{score * 100:.1f}%",
                "poster_url": poster_url
            })

            if len(recommendations) >= top_k:
                break

        return recommendations

    def get_catalogue(
        self,
        genre: Optional[str] = None,
        sort_by: str = "popularity",
        search_query: Optional[str] = None,
        limit: int = 24,
        offset: int = 0,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Browse and discover movies in the catalogue with filtering and sorting.
        Returns a page of movie records and total count.
        """
        filtered = self.df

        if genre and genre != "All Genres":
            filtered = filtered[filtered["genres_list"].apply(lambda glist: genre in glist)]

        if search_query and search_query.strip():
            q = search_query.strip().lower()
            filtered = filtered[filtered["title"].str.lower().str.contains(q, regex=False, na=False)]

        total_count = len(filtered)

        # Sorting
        if sort_by == "popularity":
            filtered = filtered.sort_values(by="popularity", ascending=False)
        elif sort_by == "vote_average":
            filtered = filtered[filtered["vote_count"] >= 50].sort_values(by="vote_average", ascending=False)
        elif sort_by == "release_year":
            filtered = filtered[filtered["release_year"] != "N/A"].sort_values(by="release_year", ascending=False)
        elif sort_by == "title":
            filtered = filtered.sort_values(by="title", ascending=True)

        page_df = filtered.iloc[offset: offset + limit]

        results = []
        for _, row in page_df.iterrows():
            mid = int(row["id"])
            s_mid = str(mid)
            poster_url = None

            # Check dataframe row poster_path
            poster_path = row["poster_path"]
            # Check posters_cache
            if not poster_path or not isinstance(poster_path, str) or not poster_path.strip():
                poster_path = self.posters_cache.get(mid) or self.posters_cache.get(s_mid)

            if poster_path and isinstance(poster_path, str) and poster_path.strip():
                if poster_path.startswith("http"):
                    poster_url = poster_path
                else:
                    poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path.strip()}"
            else:
                poster_url = self.fetch_poster_for_movie(mid, api_key=api_key)

            results.append({
                "id": mid,
                "title": row["title"],
                "release_year": row["release_year"],
                "genres": row["genres_list"],
                "genres_display": row["genres_display"],
                "vote_average": round(float(row["vote_average"]), 1),
                "vote_count": int(row["vote_count"]),
                "popularity": round(float(row["popularity"]), 1),
                "director": row["director_display"],
                "cast": row["cast_display"],
                "overview": row["overview"],
                "poster_url": poster_url
            })

        return results, total_count


    def fetch_poster_for_movie(self, movie_id: int, api_key: Optional[str] = None) -> Optional[str]:
        """Dynamically fetch poster path from TMDB API if not already cached."""
        s_mid = str(movie_id)
        if s_mid in self.posters_cache and self.posters_cache[s_mid]:
            return f"{TMDB_IMAGE_BASE_URL}{self.posters_cache[s_mid]}"

        key = api_key if api_key else "158402ec29d13b392ed3bff727897e84"
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}"
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode())
                poster_path = data.get("poster_path")
                if poster_path:
                    self.posters_cache[s_mid] = poster_path
                    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        except Exception:
            pass
        return None

    def get_movie_details(self, title_or_id: Any, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve full structured metadata for a selected movie."""
        if isinstance(title_or_id, int) or (isinstance(title_or_id, str) and str(title_or_id).isdigit()):
            row = self.get_movie_by_id(int(title_or_id))
        else:
            row = self.get_movie_by_title(str(title_or_id))

        if row is None:
            return None

        mid = int(row["id"])
        s_mid = str(mid)
        poster_url = None
        if s_mid in self.posters_cache and self.posters_cache[s_mid]:
            poster_url = f"{TMDB_IMAGE_BASE_URL}{self.posters_cache[s_mid]}"
        elif row["poster_path"] and isinstance(row["poster_path"], str) and row["poster_path"].strip():
            poster_url = f"{TMDB_IMAGE_BASE_URL}{row['poster_path'].strip()}"
        else:
            poster_url = self.fetch_poster_for_movie(mid, api_key=api_key)

        return {
            "id": mid,
            "title": row["title"],
            "release_year": row["release_year"],
            "release_date": row["release_date"],
            "genres": row["genres_list"],
            "genres_display": row["genres_display"],
            "vote_average": round(float(row["vote_average"]), 1),
            "vote_count": int(row["vote_count"]),
            "popularity": round(float(row["popularity"]), 1),
            "runtime": int(row["runtime"]) if row["runtime"] else "N/A",
            "director": row["director_display"],
            "cast": row["cast_display"],
            "overview": row["overview"],
            "tagline": row["tagline"],
            "poster_url": poster_url
        }
