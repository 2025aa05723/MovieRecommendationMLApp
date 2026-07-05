"""
utils.py - Helper functions for data filtering and processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def parse_genres(genre_string: str) -> List[str]:
    """
    Parse genre string into list of genres.
    
    Args:
        genre_string: Comma-separated genre string
        
    Returns:
        List of genres
    """
    if not genre_string:
        return []
    return [g.strip() for g in genre_string.split(',')]

def parse_cast(cast_string: str) -> List[str]:
    """
    Parse cast string into list of actors.
    
    Args:
        cast_string: Comma-separated cast string
        
    Returns:
        List of actors
    """
    if not cast_string:
        return []
    return [actor.strip() for actor in cast_string.split(',')]

def filter_movies(
    movies_df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Filter movies based on criteria.
    
    Args:
        movies_df: DataFrame containing movies
        **kwargs: Filter criteria (genre, min_rating, year_range, etc.)
        
    Returns:
        Filtered DataFrame
    """
    result = movies_df.copy()
    
    if 'genre' in kwargs and kwargs['genre']:
        result = result[result['genre'].str.contains(kwargs['genre'], case=False, na=False)]
    
    if 'min_rating' in kwargs:
        result = result[result['rating'] >= kwargs['min_rating']]
    
    if 'max_rating' in kwargs:
        result = result[result['rating'] <= kwargs['max_rating']]
    
    if 'year_from' in kwargs:
        result = result[result['year'] >= kwargs['year_from']]
    
    if 'year_to' in kwargs:
        result = result[result['year'] <= kwargs['year_to']]
    
    return result

def get_recommendations_summary(
    recommendations: List[tuple],
    movies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a summary DataFrame of recommendations.
    
    Args:
        recommendations: List of (title, score) tuples
        movies_df: DataFrame containing movie data
        
    Returns:
        DataFrame with recommendation details
    """
    data = []
    for idx, (title, score) in enumerate(recommendations, 1):
        try:
            movie = movies_df[movies_df['title'] == title].iloc[0]
            data.append({
                'rank': idx,
                'title': title,
                'similarity_score': score,
                'rating': movie['rating'],
                'genre': movie['genre'],
                'year': movie['year']
            })
        except IndexError:
            continue
    
    return pd.DataFrame(data)

def calculate_recommendation_stats(recommendations: List[tuple]) -> Dict[str, Any]:
    """
    Calculate statistics about recommendations.
    
    Args:
        recommendations: List of (title, score) tuples
        
    Returns:
        Dictionary of statistics
    """
    if not recommendations:
        return {}
    
    scores = [score for _, score in recommendations]
    
    return {
        'avg_score': np.mean(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'std_score': np.std(scores),
        'count': len(recommendations)
    }

def format_movie_info(movie_series: pd.Series) -> Dict[str, Any]:
    """
    Format movie information for display.
    
    Args:
        movie_series: Movie data as pandas Series
        
    Returns:
        Dictionary with formatted movie info
    """
    return {
        'title': movie_series.get('title', 'Unknown'),
        'rating': f"{movie_series.get('rating', 'N/A')}/10",
        'year': int(movie_series.get('year', 0)) if movie_series.get('year') else 'N/A',
        'genre': movie_series.get('genre', 'Unknown'),
        'cast': movie_series.get('cast', 'Unknown'),
        'plot': movie_series.get('plot', 'No plot available')
    }
