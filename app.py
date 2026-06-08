import streamlit as st
import pandas as pd
import numpy as np
from src.recommender import MovieRecommender
from src.data_loader import DataLoader
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background-color: #e8f0fe;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77e6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load and cache movie data and models"""
    loader = DataLoader()
    movies_df = loader.load_movies()
    recommender = MovieRecommender(movies_df)
    return movies_df, recommender

def display_movie_card(movie_info, score=None):
    """Display a movie information card"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Rating", f"{movie_info.get('rating', 'N/A')}/10")
    
    with col2:
        title = movie_info.get('title', 'Unknown')
        year = movie_info.get('year', 'N/A')
        genre = movie_info.get('genre', 'N/A')
        plot = movie_info.get('plot', 'No plot available')[:200] + '...'
        
        if score:
            st.write(f"**{title}** ({year}) - Similarity: {score:.2%}")
        else:
            st.write(f"**{title}** ({year})")
        
        st.write(f"*Genre: {genre}*")
        st.write(f"{plot}")

def main():
    # Header
    st.title("🎬 Movie Recommendation Engine")
    st.markdown("*Discover your next favorite movie using AI-powered recommendations*")
    
    # Load data
    movies_df, recommender = load_data()
    
    # Sidebar filters
    st.sidebar.header("⚙️ Filters & Settings")
    
    recommendation_type = st.sidebar.radio(
        "Recommendation Method:",
        ["Content-Based", "Collaborative", "Hybrid"],
        help="Choose how recommendations are generated"
    )
    
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    min_rating = st.sidebar.slider(
        "Minimum Movie Rating:",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5
    )
    
    # Filter movies by rating
    filtered_movies = movies_df[movies_df['rating'] >= min_rating].copy()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search & Recommend", "📊 Statistics", "🎯 Browse", "ℹ️ About"])
    
    # Tab 1: Search and Recommend
    with tab1:
        st.header("Find Movie Recommendations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_movie = st.selectbox(
                "Select a movie:",
                options=filtered_movies['title'].tolist(),
                help="Choose a movie you like to get recommendations"
            )
        
        with col2:
            if st.button("🎯 Get Recommendations", use_container_width=True):
                st.session_state.show_recommendations = True
        
        if st.session_state.get('show_recommendations', False):
            st.divider()
            
            # Get the selected movie info
            selected_movie_info = filtered_movies[filtered_movies['title'] == selected_movie].iloc[0]
            
            st.subheader(f"📌 Selected Movie: {selected_movie}")
            display_movie_card(selected_movie_info)
            
            st.divider()
            
            # Get recommendations
            with st.spinner(f"🔄 Getting {recommendation_type} recommendations..."):
                if recommendation_type == "Content-Based":
                    recommendations = recommender.content_based_recommendations(
                        selected_movie,
                        n_recommendations=num_recommendations
                    )
                elif recommendation_type == "Collaborative":
                    recommendations = recommender.collaborative_recommendations(
                        selected_movie,
                        n_recommendations=num_recommendations
                    )
                else:  # Hybrid
                    recommendations = recommender.hybrid_recommendations(
                        selected_movie,
                        n_recommendations=num_recommendations
                    )
            
            st.subheader(f"✨ Top {num_recommendations} Recommendations")
            
            for idx, (movie_title, score) in enumerate(recommendations, 1):
                movie_info = filtered_movies[filtered_movies['title'] == movie_title].iloc[0]
                st.markdown(
                    f'<div class="recommendation-card">'
                    f'<h4>#{idx} - Similarity Score: {score:.2%}</h4></div>',
                    unsafe_allow_html=True
                )
                display_movie_card(movie_info, score)
                st.divider()
    
    # Tab 2: Statistics
    with tab2:
        st.header("📊 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Movies", len(filtered_movies))
        col2.metric("Average Rating", f"{filtered_movies['rating'].mean():.2f}")
        col3.metric("Genres", filtered_movies['genre'].nunique())
        col4.metric("Years Covered", f"{filtered_movies['year'].min():.0f} - {filtered_movies['year'].max():.0f}")
        
        st.divider()
        
        # Rating distribution
        fig_rating = px.histogram(
            filtered_movies,
            x='rating',
            nbins=20,
            title='Rating Distribution',
            labels={'rating': 'Rating', 'count': 'Number of Movies'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_rating, use_container_width=True)
        
        # Top genres
        genre_counts = filtered_movies['genre'].value_counts().head(10)
        fig_genre = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            title='Top 10 Genres',
            labels={'x': 'Number of Movies', 'y': 'Genre'},
            color_discrete_sequence=['#EF553B']
        )
        fig_genre.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_genre, use_container_width=True)
        
        # Movies by year
        fig_year = px.line(
            filtered_movies.groupby('year').size().reset_index(name='count'),
            x='year',
            y='count',
            title='Movies by Release Year',
            labels={'year': 'Year', 'count': 'Number of Movies'},
            markers=True,
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig_year, use_container_width=True)
    
    # Tab 3: Browse
    with tab3:
        st.header("🎯 Browse Movies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_genre = st.selectbox(
                "Filter by Genre:",
                options=["All"] + sorted(filtered_movies['genre'].unique().tolist())
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                options=["Rating (High to Low)", "Rating (Low to High)", "Year (Newest)", "Year (Oldest)"]
            )
        
        with col3:
            items_per_page = st.selectbox(
                "Items per page:",
                options=[5, 10, 20, 50]
            )
        
        # Filter by genre
        if selected_genre != "All":
            browse_movies = filtered_movies[filtered_movies['genre'] == selected_genre].copy()
        else:
            browse_movies = filtered_movies.copy()
        
        # Sort
        if sort_by == "Rating (High to Low)":
            browse_movies = browse_movies.sort_values('rating', ascending=False)
        elif sort_by == "Rating (Low to High)":
            browse_movies = browse_movies.sort_values('rating', ascending=True)
        elif sort_by == "Year (Newest)":
            browse_movies = browse_movies.sort_values('year', ascending=False)
        else:
            browse_movies = browse_movies.sort_values('year', ascending=True)
        
        # Display movies
        for idx, (_, movie) in enumerate(browse_movies.head(items_per_page).iterrows(), 1):
            st.markdown(f"### {idx}. {movie['title']} ({int(movie['year'])})")
            col1, col2, col3 = st.columns([1, 2, 1])
            col1.metric("Rating", f"{movie['rating']}/10")
            col2.write(f"*{movie['genre']}*")
            col3.metric("Year", int(movie['year']))
            st.divider()
    
    # Tab 4: About
    with tab4:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### How It Works
        
        This movie recommendation engine uses multiple machine learning algorithms:
        
        **1. Content-Based Filtering**
        - Analyzes movie features (genre, cast, director, plot)
        - Finds movies similar to your selection
        - Uses TF-IDF vectorization and cosine similarity
        
        **2. Collaborative Filtering**
        - Studies user rating patterns
        - Finds users with similar preferences
        - Recommends movies liked by similar users
        
        **3. Hybrid Approach**
        - Combines both methods
        - Provides more accurate recommendations
        - Weights factors from both algorithms
        
        ### Technologies
        
        - **Streamlit**: Interactive web application
        - **scikit-learn**: Machine learning algorithms
        - **Pandas & NumPy**: Data processing
        - **Plotly**: Interactive visualizations
        
        ### Dataset
        
        This application uses movie data from:
        - MovieLens dataset
        - IMDb ratings and metadata
        - Genre and cast information
        """)
        
        st.divider()
        
        st.markdown("""
        ### Features
        
        ✅ Multiple recommendation algorithms  
        ✅ Real-time filtering and sorting  
        ✅ Interactive visualizations  
        ✅ Detailed movie information  
        ✅ Customizable recommendation count  
        ✅ Rating-based filtering  
        
        ### Future Enhancements
        
        - Deep learning models
        - User account system
        - Personalized recommendations
        - Social features
        - API endpoints
        """)

if __name__ == "__main__":
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    main()
