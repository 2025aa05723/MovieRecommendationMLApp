from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from src.repository import MovieRepository, CSVMovieRepository
from src.strategies import (
    RecommendationStrategy,
    ContentBasedStrategy,
    CollaborativeStrategy,
    HybridStrategy
)

class RecommendationService:
    """
    Application Service layer that coordinates the presentation layer with the
    underlying Data Repositories and Recommendation Strategies.
    """
    
    def __init__(self, movie_repo: Optional[MovieRepository] = None):
        # Default to CSV implementation if no repository is passed (Dependency Injection support)
        self.movie_repo = movie_repo if movie_repo is not None else CSVMovieRepository()
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Fetch preprocessed dataset from repository and construct strategies."""
        movies_df = self.movie_repo.get_all_movies()
        
        self.content_strategy = ContentBasedStrategy(movies_df)
        self.collaborative_strategy = CollaborativeStrategy(movies_df)
        self.hybrid_strategy = HybridStrategy(self.content_strategy, self.collaborative_strategy)
        
        self.strategies: Dict[str, RecommendationStrategy] = {
            "Content-Based": self.content_strategy,
            "Collaborative": self.collaborative_strategy,
            "Hybrid": self.hybrid_strategy
        }

    def get_recommendations(
        self,
        method: str,
        movie_title: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations using the selected strategy.
        
        Args:
            method: Name of the strategy ('Content-Based', 'Collaborative', or 'Hybrid')
            movie_title: Title of the base movie
            n_recommendations: Number of recommendations to return
        """
        strategy = self.strategies.get(method)
        if not strategy:
            raise ValueError(f"Unknown recommendation strategy method: {method}")
        return strategy.recommend(movie_title, n_recommendations)

    def get_similar_movies(
        self,
        movie_title: str,
        n_movies: int = 5,
        method: str = 'content'
    ) -> List[Tuple[str, float]]:
        """Compatibility helper matching the legacy recommender signature."""
        method_map = {
            'content': 'Content-Based',
            'collab': 'Collaborative',
            'hybrid': 'Hybrid'
        }
        mapped_method = method_map.get(method.lower(), 'Content-Based')
        return self.get_recommendations(mapped_method, movie_title, n_movies)

    def get_all_movies(self) -> pd.DataFrame:
        """Get all preprocessed movies from the repository."""
        return self.movie_repo.get_all_movies()

    def get_movie_info(self, movie_title: str) -> Optional[pd.Series]:
        """Fetch details for a single movie by title."""
        return self.movie_repo.get_movie_by_title(movie_title)

    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search the repository for movies matching query in title."""
        movies = self.get_all_movies()
        mask = movies['title'].str.contains(query, case=False, na=False)
        return movies[mask].head(limit)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return high-level statistics about the movie dataset."""
        df = self.get_all_movies()
        return {
            'total_movies': len(df),
            'avg_rating': float(df['rating'].mean()) if not df.empty else 0.0,
            'min_rating': float(df['rating'].min()) if not df.empty else 0.0,
            'max_rating': float(df['rating'].max()) if not df.empty else 0.0,
            'total_genres': int(df['genre'].nunique()),
            'year_range': f"{int(df['year'].min())} - {int(df['year'].max())}" if not df.empty else "N/A"
        }
