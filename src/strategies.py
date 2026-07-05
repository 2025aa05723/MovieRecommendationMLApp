import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationStrategy(ABC):
    """
    Abstract interface for recommendation algorithms.
    """
    @abstractmethod
    def recommend(self, movie_title: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a movie title.
        
        Args:
            movie_title: Title of the base movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples containing (movie_title, similarity_score)
        """
        pass

class ContentBasedStrategy(RecommendationStrategy):
    """
    Content-Based Recommendation Strategy using TF-IDF and Cosine Similarity.
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        self._initialize_tfidf()

    def _initialize_tfidf(self) -> None:
        # Combine text features
        self.movies_df['combined_features'] = (
            self.movies_df['genre'].fillna('') + ' ' +
            self.movies_df['cast'].fillna('') + ' ' +
            self.movies_df['plot'].fillna('')
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['combined_features']
        )
        
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_title: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        try:
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        except IndexError:
            return []
            
        sim_scores = self.similarity_matrix[movie_idx]
        
        # Get top recommendations (excluding the movie itself)
        top_indices = np.argsort(sim_scores)[::-1][1:n_recommendations + 1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append((
                self.movies_df.iloc[idx]['title'],
                float(sim_scores[idx])
            ))
            
        return recommendations

class CollaborativeStrategy(RecommendationStrategy):
    """
    Collaborative Recommendation Strategy using Genre Overlap and Rating Proximity.
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)

    def recommend(self, movie_title: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        try:
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
            selected_movie = self.movies_df.iloc[movie_idx]
        except IndexError:
            return []
            
        # Find movies with similar genres (matching the primary genre)
        primary_genre = selected_movie['genre'].split(',')[0].strip() if selected_movie['genre'] else ""
        
        if not primary_genre:
            genre_matches = self.movies_df.copy()
        else:
            genre_matches = self.movies_df[
                self.movies_df['genre'].str.contains(primary_genre, case=False, na=False)
            ].copy()
            
        # Calculate rating similarity
        genre_matches['rating_diff'] = abs(
            genre_matches['rating'] - selected_movie['rating']
        )
        
        # Proximity score calculation
        genre_matches['score'] = 1 / (1 + genre_matches['rating_diff'])
        
        # Exclude self
        genre_matches = genre_matches[genre_matches.index != movie_idx]
        
        top_movies = genre_matches.nlargest(n_recommendations, 'score')
        
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append((
                movie['title'],
                float(movie['score'])
            ))
            
        return recommendations

class HybridStrategy(RecommendationStrategy):
    """
    Hybrid Recommendation Strategy combining Content-Based and Collaborative Filtering.
    """
    def __init__(self, content_strategy: ContentBasedStrategy, collaborative_strategy: CollaborativeStrategy):
        self.content_strategy = content_strategy
        self.collaborative_strategy = collaborative_strategy

    def recommend(self, movie_title: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        # Request double recommendations to ensure overlap and selection diversity
        content_recs = self.content_strategy.recommend(movie_title, n_recommendations * 2)
        collab_recs = self.collaborative_strategy.recommend(movie_title, n_recommendations * 2)
        
        # Blend and apply default weights (60% Content-based, 40% Collaborative)
        content_weight = 0.6
        collab_weight = 0.4
        
        scores = {}
        for movie, score in content_recs:
            scores[movie] = scores.get(movie, 0.0) + score * content_weight
            
        for movie, score in collab_recs:
            scores[movie] = scores.get(movie, 0.0) + score * collab_weight
            
        if not scores:
            return []
            
        # Normalize scores
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
            
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
