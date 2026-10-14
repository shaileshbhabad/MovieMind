"""
test_recommender.py
Test suite to validate all ML and functional requirements for MovieMind.
"""

import time
from recommender import MovieRecommender


def run_tests():
    print("--- 1. Testing Model Initialization ---")
    t0 = time.time()
    rec = MovieRecommender()
    init_time = time.time() - t0
    print(f"Model initialized in {init_time:.2f} seconds.")
    print(f"Total movies loaded: {len(rec.df)}")
    assert len(rec.df) == 4803, f"Expected 4803 movies, got {len(rec.df)}"

    print("\n--- 2. Testing Movie Search ---")
    results = rec.search_titles("Dark Knight", limit=5)
    print("Search results for 'Dark Knight':", results)
    assert any("The Dark Knight" in t for t in results), "Failed to find 'The Dark Knight'"

    print("\n--- 3. Testing Recommendations & Self-Exclusion ---")
    test_movies = ["Avatar", "The Dark Knight", "Inception", "Toy Story"]
    for title in test_movies:
        t_rec = time.time()
        recommendations = rec.get_recommendations(title, top_k=5)
        rec_duration_ms = (time.time() - t_rec) * 1000

        print(f"\nRecommendations for '{title}' (generated in {rec_duration_ms:.2f} ms):")
        assert len(recommendations) == 5, f"Expected 5 recommendations, got {len(recommendations)}"

        for r in recommendations:
            # Self-exclusion check
            assert r["title"].lower() != title.lower(), f"Self-exclusion failed: {r['title']} matches query {title}"
            print(f"  • {r['title']} ({r['release_year']}) - Similarity: {r['similarity_percentage']} | Genres: {r['genres_display']} | Director: {r['director']}")

    print("\n--- 4. Testing Genre Filtering ---")
    genre_test = "Animation"
    anim_recs = rec.get_recommendations("Avatar", top_k=5, genre_filter=genre_test)
    print(f"Avatar recommendations filtered by genre '{genre_test}':")
    for r in anim_recs:
        assert genre_test in r["genres"], f"Genre filter failed: {genre_test} not in {r['genres']}"
        print(f"  • {r['title']} | Genres: {r['genres_display']} | Similarity: {r['similarity_percentage']}")

    print("\n--- 5. Testing Empty Results Handling ---")
    empty_recs = rec.get_recommendations("NonExistentMovie12345", top_k=5)
    assert empty_recs == [], "Expected empty list for non-existent movie"
    print("Non-existent movie correctly returned empty list.")

    print("\n--- 6. Testing Catalogue Explorer ---")
    catalogue, total = rec.get_catalogue(genre="Action", sort_by="vote_average", limit=4)
    print(f"Catalogue query for Action movies: total {total} movies found. First 4:")
    for m in catalogue:
        print(f"  • {m['title']} | Rating: {m['vote_average']} (Votes: {m['vote_count']})")
    assert total > 0, "Catalogue returned 0 movies"

    print("\n--- 7. Evaluation Metrics Check ---")
    # Genre relevance metric: % of top 5 recommendations sharing at least 1 genre with query movie
    total_evaluated = 0
    relevant_genre_matches = 0
    sample_movies = ["The Avengers", "Interstellar", "Finding Nemo", "The Godfather", "The Matrix"]

    for sm in sample_movies:
        orig = rec.get_movie_by_title(sm)
        if orig is None:
            continue
        orig_genres = set(orig["genres_list"])
        recs = rec.get_recommendations(sm, top_k=5)
        for r in recs:
            total_evaluated += 1
            if orig_genres.intersection(set(r["genres"])):
                relevant_genre_matches += 1

    genre_relevance_rate = (relevant_genre_matches / total_evaluated) * 100 if total_evaluated > 0 else 0
    print(f"Genre Relevance @ 5 across sample films: {genre_relevance_rate:.1f}% ({relevant_genre_matches}/{total_evaluated})")
    assert genre_relevance_rate >= 80.0, f"Genre relevance should be >= 80%, got {genre_relevance_rate:.1f}%"

    print("\nALL TESTS PASSED SUCCESSFULLY! Model and engine are fully operational.")


if __name__ == "__main__":
    run_tests()
