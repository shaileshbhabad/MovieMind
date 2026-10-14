# 🎬 MovieMind — Personalized Movie Recommendation System

MovieMind is a modern Machine Learning web application that delivers personalized film recommendations using **Content-Based Filtering**, **Natural Language Processing (NLP)**, and **Cosine Similarity**. Built on the TMDB 5000 Movie Dataset, the platform analyzes plot summaries, genre classifications, directors, leading actors, and thematic keywords to calculate mathematically ranked movie recommendations.

---

## 🌟 Key Features

- **Genuine Machine Learning Engine:** Powered by Scikit-learn's `TfidfVectorizer` (unigrams + bigrams, 20,000 features) and linear cosine similarity over real TMDB metadata.
- **Cinematic Dark Theme:** Premium streaming-style UI designed with deep charcoals (`#0B0D14`, `#151824`), violet/electric-purple accents (`#9B72F2`), and clean responsive cards.
- **Instant Interactive Recommendations:** Sub-25ms recommendation calculation via sparse matrix operations and cached model state (`@st.cache_resource`).
- **Dynamic Genre Filtering:** Filter recommendations on-the-fly across 20 distinct TMDB genres with automatic fallback and reset capabilities.
- **Movie Spotlight & Pivoting:** View deep movie metadata (overview, director, cast, runtime, rating, release year, tagline) and pivot into recommendations from any suggested film with a single click.
- **Movie Explorer:** Browse and search the 4,803-film catalogue by title, genre, and sort by popularity, rating, or release year.
- **High-Resolution TMDB Posters:** 800+ popular movie posters pre-cached offline with high-speed TMDB API fallback and custom SVG vector placeholders for missing artwork.
- **Interactive Similarity Analyzer:** Compare any two movies to inspect their mathematical cosine similarity, shared genres, and director matches.

---

## 🏗️ Technical Architecture

MovieMind is organized into three logical tiers:

```
┌────────────────────────────────────────────────────────┐
│                   Presentation Layer                   │
│          Streamlit Web Application (app.py)            │
│  - Cinematic Dark UI, Responsive Grids & Card Badges   │
│  - Interactive Search, Genre Filters & Sliders         │
│  - Movie Explorer & Real-Time Similarity Analyzer      │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│                 Machine Learning Layer                 │
│              Recommender (recommender.py)              │
│  - Metadata Feature Soup (Genres, Plot, Crew, Cast)    │
│  - TF-IDF Vectorizer (20,000 unigrams + bigrams)       │
│  - Cosine Similarity & Self-Exclusion Ranking          │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│                       Data Layer                       │
│                   TMDB 5000 Dataset                    │
│  - tmdb_5000_movies.csv (4,803 movies, 20 attributes)  │
│  - tmdb_5000_credits.csv (cast, crew, directors)       │
│  - posters_cache.json (800+ offline poster URLs)       │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Methodology

### 1. Feature Engineering (The Metadata "Soup")
For each film in the TMDB catalogue, textual metadata is extracted, cleaned, and combined:
- **Plot Overview & Tagline:** Semantic description of the movie's narrative.
- **Weighted Genres ($2\times$):** Doubled to increase genre adherence in recommendation similarity.
- **Thematic Keywords ($2\times$):** Core motifs (e.g., "space war", "dystopia", "time travel").
- **Director Signature:** Extracted from crew data and tokenized as a single word (e.g., `Christopher Nolan` $\rightarrow$ `christophernolan`).
- **Prominent Cast:** Top 4 billed actors tokenized into distinct entity strings to avoid cross-matching unrelated individuals sharing common first names.

### 2. TF-IDF Vectorization
The combined text representation is converted into numerical feature vectors using `TfidfVectorizer`:
- **N-gram Range:** Unigrams and bigrams `(1, 2)` to capture phrases (e.g., "science fiction", "dark knight").
- **Stop Words:** English stop words removed.
- **Max Features:** 20,000 most significant terms.
- **Sublinear TF Scaling:** Logarithmic term frequency dampening.

### 3. Cosine Similarity Ranking
Because TF-IDF vectors are $L_2$-normalized, the cosine similarity between the selected movie vector $u$ and candidate vector $v$ equals their dot product:

$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$

The system excludes the query movie itself, applies genre filters if requested, and returns the top $K$ results ranked by similarity score.

---

## 📁 Project Directory Structure

```text
MovieMind/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── recommender.py             # Core ML recommendation engine & data pipeline
├── ui_components.py           # Custom CSS, card templates, and SVG placeholders
├── test_recommender.py        # Automated test suite and validation scripts
├── build_poster_cache.py      # Multi-threaded TMDB poster caching utility
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── tmdb_5000_movies.csv   # TMDB 5000 movies metadata
│   ├── tmdb_5000_credits.csv  # TMDB 5000 cast and crew metadata
│   └── posters_cache.json     # 800+ pre-cached official TMDB posters
│
├── models/                    # Directory for optional saved models
└── assets/                    # Optional graphics and static assets
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- Python 3.10 or Python 3.11
- Internet connection (for initial setup and TMDB images)

### 2. Clone or Navigate to the Project
```bash
cd "MovieMind"
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Automated Tests
Verify that dataset loading, vectorization, self-exclusion, and recommendation quality pass all assertions:
```bash
python test_recommender.py
```

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use MovieMind.

---

## 🧪 Testing & Validation Results

The automated test suite (`test_recommender.py`) validates:

| Test Case | Scenario | Status | Measured Metric |
| :--- | :--- | :---: | :--- |
| **Test 1: Startup & Ingestion** | Both CSVs merged on `movie_id` | ✅ Passed | 4,803 films loaded in 6.2s |
| **Test 2: Title Search** | Prefix & substring case-insensitive lookup | ✅ Passed | Instant matching |
| **Test 3: Recommendation Quality** | Top matches for *The Dark Knight*, *Avatar*, *Inception* | ✅ Passed | Christopher Nolan & James Cameron films correctly ranked |
| **Test 4: Self-Exclusion** | Query movie omitted from recommendation results | ✅ Passed | 100% self-exclusion rate |
| **Test 5: Genre Filtering** | All returned recommendations contain selected genre | ✅ Passed | 100% adherence to active filter |
| **Test 6: Empty Results** | Graceful fallback when non-existent titles or zero genre matches occur | ✅ Passed | Friendly UI notification and filter reset |
| **Test 7: Genre Relevance @ 5** | Proportion of top 5 recommendations sharing $\ge 1$ genre | ✅ Passed | **88.0%** across benchmark films |
| **Test 8: Query Latency** | Sparse matrix dot-product computation | ✅ Passed | **17–28 milliseconds** per query |

---

## 📊 Evaluation Metrics

1. **Genre Relevance @ K:** Measures whether recommended movies belong to the expected thematic categories of the seed movie. MovieMind achieves **88.0%** relevance on benchmark samples.
2. **Catalogue Coverage:** MovieMind achieves **100% coverage** ($4,803 / 4,803$ movies), meaning any movie in the catalogue can be used as a recommendation seed.
3. **Sparse Execution Speed:** Precomputing and caching the TF-IDF representation ensures instant user interactions without re-training the vectorizer.

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Content-Based Scope:** Recommends movies based on metadata similarities, without individual user rating histories or collaborative community trends.
- **Metadata Cold-Start:** Highly obscure films with minimal plot overviews or missing cast details produce lower-dimensional signals.
- **Content Similarity $\neq$ Viewer Satisfaction:** Similarity scores reflect attribute overlap, not necessarily the probability that an individual viewer will enjoy the film.

### Future Roadmap
- **Collaborative Filtering:** Introduce matrix factorization (SVD / ALS) to combine user rating patterns with content features.
- **Deep Neural Embeddings:** Evaluate transformer-based models (such as Sentence-BERT) for deeper narrative semantic matching.
- **Personalized Watchlists & Trailers:** Add user accounts, custom watchlists, and embedded TMDB video trailer players.

---

## 📜 License & Acknowledgments
- Dataset provided by **The Movie Database (TMDB)** via Kaggle.
- Developed with Python, Pandas, NumPy, Scikit-learn, and Streamlit.
