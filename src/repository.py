"""
repository.py - Data access layer for movie data.

Implements the Repository Pattern:
- MovieRepository (ABC) defines the data access interface.
- CSVMovieRepository is the concrete CSV-backed implementation.

This lets the service layer stay independent of the underlying storage.
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class MovieRepository(ABC):
    """Abstract base class defining the interface for movie data access."""

    @abstractmethod
    def get_all_movies(self) -> pd.DataFrame:
        """Retrieve all movies."""
        pass

    @abstractmethod
    def get_movie_by_title(self, title: str) -> Optional[pd.Series]:
        """Retrieve a specific movie by its title."""
        pass


class CSVMovieRepository(MovieRepository):
    """Concrete implementation reading from / writing to a CSV file."""

    def __init__(self, data_dir: str = "data", filename: str = "movies.csv"):
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, filename)
        self._movies_df: Optional[pd.DataFrame] = None
        self._load_and_preprocess()

    def _load_and_preprocess(self) -> None:
        """Load data from CSV, or create sample data if the file is missing."""
        if not os.path.exists(self.file_path):
            df = self._create_sample_data()
        else:
            df = pd.read_csv(self.file_path)
        self._movies_df = self._preprocess(df)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardise the raw movie dataset."""
        df = df.copy()

        text_columns = ['genre', 'cast', 'plot']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').str.lower().str.strip()

        df = df.drop_duplicates(subset=['title'])

        # ensure required columns exist
        required_columns = ['title', 'genre', 'rating', 'year', 'cast', 'plot']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        return df.reset_index(drop=True)

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample movie data for demonstration."""
        sample_data = {
            'title': [
                'Inception', 'The Matrix', 'Interstellar', 'The Dark Knight', 'Forrest Gump',
                'Pulp Fiction', 'Fight Club', 'The Shawshank Redemption', 'Gladiator', 'Avatar',
                'Titanic', 'The Avengers', 'The Lion King', 'Jurassic Park',
                'The Lord of the Rings', 'Harry Potter', 'Star Wars', 'The Godfather', 'Casablanca',
                'Citizen Kane', 'Vertigo', 'Psycho', 'Jaws', 'E.T.',
                'Back to the Future', 'The Terminator', 'Alien', 'Blade Runner', 'The Sixth Sense'
            ],
            'genre': [
                'Sci-Fi, Thriller', 'Sci-Fi, Action', 'Sci-Fi, Drama', 'Action, Crime', 'Drama, Comedy',
                'Crime, Drama', 'Drama, Thriller', 'Drama', 'Action, Drama', 'Sci-Fi, Fantasy',
                'Romance, Drama', 'Action, Adventure', 'Animation, Adventure', 'Action, Adventure',
                'Fantasy, Adventure', 'Fantasy, Adventure', 'Sci-Fi, Adventure', 'Crime, Drama', 'Drama, Romance',
                'Drama, Mystery', 'Thriller, Mystery', 'Horror, Thriller', 'Thriller, Adventure', 'Sci-Fi, Family',
                'Comedy, Adventure', 'Action, Sci-Fi', 'Action, Horror', 'Sci-Fi, Thriller', 'Drama, Mystery'
            ],
            'rating': [
                8.8, 8.7, 8.6, 9.0, 8.8,
                8.9, 8.8, 9.3, 8.5, 7.8,
                7.2, 8.0, 8.2, 8.2,
                9.0, 7.6, 8.6, 9.2, 8.5,
                8.3, 8.3, 8.4, 8.0, 7.9,
                8.5, 8.1, 8.5, 8.1, 8.2
            ],
            'year': [
                2010, 1999, 2014, 2008, 1994,
                1994, 1999, 1994, 2000, 2009,
                1997, 2012, 1994, 1993,
                2001, 2001, 1977, 1972, 1942,
                1941, 1958, 1960, 1975, 1982,
                1985, 1984, 1979, 1982, 1999
            ],
            'cast': [
                'Leonardo DiCaprio, Ellen Page', 'Keanu Reeves, Laurence Fishburne', 'Matthew McConaughey, Anne Hathaway', 'Christian Bale, Heath Ledger', 'Tom Hanks, Gary Sinise',
                'John Travolta, Samuel L. Jackson', 'Brad Pitt, Edward Norton', 'Tim Robbins, Morgan Freeman', 'Russell Crowe, Joaquin Phoenix', 'Sam Worthington, Zoe Saldana',
                'Leonardo DiCaprio, Kate Winslet', 'Robert Downey Jr., Chris Evans', 'James Earl Jones, Jeremy Irons', 'Sam Neill, Laura Dern',
                'Elijah Wood, Ian McKellen', 'Daniel Radcliffe, Emma Watson', 'Mark Hamill, Harrison Ford', 'Marlon Brando, Al Pacino', 'Humphrey Bogart, Ingrid Bergman',
                'Orson Welles', 'James Stewart, Kim Novak', 'Anthony Perkins, Vera Miles', 'Roy Scheider, Robert Shaw', 'Henry Thomas, Dee Wallace',
                'Michael J. Fox, Christopher Lloyd', 'Arnold Schwarzenegger, Linda Hamilton', 'Sigourney Weaver, Tom Skerritt', 'Harrison Ford, Rutger Hauer', 'Bruce Willis, Haley Joel Osment'
            ],
            'plot': [
                'A thief who steals corporate secrets through dream-sharing technology', 'A hacker learns about reality from rebels fighting intelligent machines', 'A team travels through a wormhole to save humanity', 'Batman faces a criminal mastermind', 'Simple man achieves success in a complex world',
                'Lives intertwine in Los Angeles', 'An underground fight club challenges consumerism', 'Two imprisoned men bond over decades', 'A gladiator seeks revenge', 'A paraplegic marine on an alien world',
                'A poor boy and wealthy girl fall in love on a doomed ship', 'Superheroes band together to fight an alien invasion', 'Lion prince seeks revenge on his uncle', 'Explorers visit an island with prehistoric creatures',
                'Hobbits journey to destroy a powerful ring', 'Young wizard attends magic school', 'Heroes fight an evil empire', 'A mafia boss transfers control of his clandestine empire', 'A cynical nightclub owner fights for justice',
                "A newspaper owner investigates a young man's rise", 'A detective obsesses over a mysterious woman', 'A motel keeper spies on guests', 'A beach town faces giant sharks', 'An alien befriends a lonely boy',
                'A teen travels through time in a DeLorean', 'A cyborg travels back in time', 'A space crew encounters a deadly alien', 'A bounty hunter tracks androids', 'A boy sees dead people'
            ]
        }

        df = pd.DataFrame(sample_data)
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_csv(self.file_path, index=False)
        return df

    def get_all_movies(self) -> pd.DataFrame:
        """Returns the preprocessed DataFrame of all movies."""
        if self._movies_df is None:
            self._load_and_preprocess()
        return self._movies_df

    def get_movie_by_title(self, title: str) -> Optional[pd.Series]:
        """Returns a movie record as a Series, or None if not found."""
        movies = self.get_all_movies()
        matches = movies[movies['title'] == title]
        if not matches.empty:
            return matches.iloc[0]
        return None
