import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import pickle
import os

class MovieRecommender:
    """
    Movie recommendation engine using multiple algorithms.
    """
    
    def __init__(self, movies_df: pd.DataFrame, model_dir: str = "models"):
        """
        Initialize the recommender with movie data.
        
        Args:
            movies_df: DataFrame containing movie information
            model_dir: Directory to save/load models
        """
        self.movies_df = movies_df.reset_index(drop=True)
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize or load models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize TF-IDF vectorizer and compute similarity matrices.
        """
        # Combine relevant text features for content-based filtering
        self.movies_df['combined_features'] = (
            self.movies_df['genre'].fillna('') + ' ' +
            self.movies_df['cast'].fillna('') + ' ' +
            self.movies_df['plot'].fillna('')
        )
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['combined_features']
        )
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
    
    def content_based_recommendations(
        self,
        movie_title: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations for a given movie.
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (movie_title, similarity_score)
        """
        # Find movie index
        try:
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        except IndexError:
            return []
        
        # Get similarity scores
        sim_scores = self.similarity_matrix[movie_idx]
        
        # Get top recommendations (excluding the movie itself)
        top_indices = np.argsort(sim_scores)[::-1][1:n_recommendations + 1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append((
                self.movies_df.iloc[idx]['title'],
                sim_scores[idx]
            ))
        
        return recommendations
    
    def collaborative_recommendations(
        self,
        movie_title: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get collaborative filtering recommendations.
        Uses genre and rating similarity.
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (movie_title, similarity_score)
        """
        try:
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
            selected_movie = self.movies_df.iloc[movie_idx]
        except IndexError:
            return []
        
        # Find movies with similar genres
        genre_matches = self.movies_df[
            self.movies_df['genre'].str.contains(
                selected_movie['genre'].split(',')[0],
                case=False,
                na=False
            )
        ].copy()
        
        # Calculate rating similarity
        genre_matches['rating_diff'] = abs(
            genre_matches['rating'] - selected_movie['rating']
        )
        
        # Score based on rating proximity
        genre_matches['score'] = 1 / (1 + genre_matches['rating_diff'])
        
        # Remove the selected movie itself
        genre_matches = genre_matches[genre_matches.index != movie_idx]
        
        # Sort and get top recommendations
        top_movies = genre_matches.nlargest(n_recommendations, 'score')
        
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append((movie['title'], movie['score']))
        
        return recommendations
    
    def hybrid_recommendations(
        self,
        movie_title: str,
        n_recommendations: int = 5,
        content_weight: float = 0.6,
        collab_weight: float = 0.4
    ) -> List[Tuple[str, float]]:
        """
        Get hybrid recommendations combining content-based and collaborative filtering.
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            content_weight: Weight for content-based recommendations
            collab_weight: Weight for collaborative recommendations
            
        Returns:
            List of tuples (movie_title, similarity_score)
        """
        # Get recommendations from both methods
        content_recs = self.content_based_recommendations(movie_title, n_recommendations * 2)
        collab_recs = self.collaborative_recommendations(movie_title, n_recommendations * 2)
        
        # Create score dictionary
        scores = {}
        
        for movie, score in content_recs:
            scores[movie] = scores.get(movie, 0) + score * content_weight
        
        for movie, score in collab_recs:
            scores[movie] = scores.get(movie, 0) + score * collab_weight
        
        # Normalize and sort
        if scores:
            max_score = max(scores.values())
            scores = {k: v / max_score for k, v in scores.items()}
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def get_similar_movies(
        self,
        movie_title: str,
        n_movies: int = 5,
        method: str = 'content'
    ) -> List[Tuple[str, float]]:
        """
        Get similar movies using specified method.
        
        Args:
            movie_title: Title of the movie
            n_movies: Number of similar movies to return
            method: 'content', 'collab', or 'hybrid'
            
        Returns:
            List of tuples (movie_title, similarity_score)
        """
        if method == 'collab':
            return self.collaborative_recommendations(movie_title, n_movies)
        elif method == 'hybrid':
            return self.hybrid_recommendations(movie_title, n_movies)
        else:  # default to content
            return self.content_based_recommendations(movie_title, n_movies)
    
    def get_movie_info(self, movie_title: str) -> Optional[pd.Series]:
        """
        Get detailed information about a movie.
        
        Args:
            movie_title: Title of the movie
            
        Returns:
            Series with movie information or None if not found
        """
        try:
            return self.movies_df[self.movies_df['title'] == movie_title].iloc[0]
        except IndexError:
            return None
    
    def search_movies(
        self,
        query: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Search for movies by title or other attributes.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.movies_df['title'].str.contains(query, case=False, na=False)
        return self.movies_df[mask].head(limit)
    
    def save_models(self):
        """
        Save trained models to disk.
        """
        with open(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(os.path.join(self.model_dir, 'similarity_matrix.pkl'), 'wb') as f:
            pickle.dump(self.similarity_matrix, f)
    
    def load_models(self):
        """
        Load trained models from disk.
        """
        try:
            with open(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            with open(os.path.join(self.model_dir, 'similarity_matrix.pkl'), 'rb') as f:
                self.similarity_matrix = pickle.load(f)
        except FileNotFoundError:
            print("Models not found. Please train models first.")
