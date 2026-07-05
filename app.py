"""
Movie Recommendation Application - Streamlit Frontend

SEML Assignment: Demonstrates layered architecture with:
- Repository Pattern (repository.py) - data access abstraction
- Strategy Pattern (strategies.py) - swappable recommendation algorithms
- Service Layer (service.py) - orchestrates repository + strategies

Group: 72
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.service import RecommendationService
import plotly.express as px
import plotly.graph_objects as go

# page setup
st.set_page_config(
    page_title="Movie Recommender",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    /* light-blue primary button */
    .stButton > button[kind="primary"] {
        background-color: #4a90e2;
        border-color: #4a90e2;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #357ABD;
        border-color: #357ABD;
        color: white;
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
def load_service():
    """Load and cache the recommendation service."""
    return RecommendationService()


def show_movie_card(movie_info, score=None):
    """Display movie details in a card layout."""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Rating", f"{movie_info.get('rating', 'N/A')}/10")
    with col2:
        title = movie_info.get('title', 'Unknown')
        year = movie_info.get('year', 'N/A')
        genre = movie_info.get('genre', 'N/A')
        plot = str(movie_info.get('plot', 'No plot available'))[:200]

        if score:
            st.write(f"**{title}** ({year}) - Similarity: {score:.2%}")
        else:
            st.write(f"**{title}** ({year})")
        st.write(f"*Genre: {genre}*")
        st.write(plot)


def main():
    st.title(":red[:material/movie:] Movie Recommendation Engine")
    st.markdown("*AI-powered movie recommendations using multiple ML strategies*")

    # load service (cached)
    service = load_service()
    movies_df = service.get_all_movies()

    # --- Sidebar ---
    st.sidebar.header(":gray[:material/settings:] Settings")

    rec_type = st.sidebar.radio(
        "Recommendation Method:",
        ["Content-Based", "Collaborative", "Hybrid"],
        help="Select which algorithm to use"
    )

    num_recs = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

    min_rating = st.sidebar.slider("Minimum Rating Filter:", 0.0, 10.0, 6.0, 0.5)

    # filter by rating
    filtered_df = movies_df[movies_df['rating'] >= min_rating].copy()

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        ":blue[:material/search:] Recommend",
        ":green[:material/bar_chart:] Statistics",
        ":orange[:material/grid_view:] Browse",
        ":gray[:material/info:] About"
    ])

    # ===== TAB 1: Recommend =====
    with tab1:
        st.header("Get Movie Recommendations")

        col1, col2 = st.columns([3, 2])
        with col1:
            selected_movie = st.selectbox(
                "Pick a movie you like:",
                options=filtered_df['title'].tolist()
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            get_recs = st.button(
                ":material/recommend: Get Recommendations",
                use_container_width=True,
                type="primary"
            )

        if get_recs:
            st.session_state.show_recs = True

        if st.session_state.get('show_recs', False) and selected_movie:
            st.divider()

            # show selected movie
            sel_info = filtered_df[filtered_df['title'] == selected_movie].iloc[0]
            st.subheader(f":red[:material/push_pin:] Selected: {selected_movie}")
            show_movie_card(sel_info)
            st.divider()

            # call service to get recommendations
            with st.spinner(f"Computing {rec_type} recommendations..."):
                recommendations = service.get_recommendations(
                    method=rec_type,
                    movie_title=selected_movie,
                    n_recommendations=num_recs
                )

            st.subheader(f":orange[:material/auto_awesome:] Top {num_recs} Recommendations ({rec_type})")

            if not recommendations:
                st.warning("No recommendations found for this movie.")
            else:
                for rank, (title, score) in enumerate(recommendations, 1):
                    try:
                        movie_info = filtered_df[filtered_df['title'] == title].iloc[0]
                    except IndexError:
                        movie_info = movies_df[movies_df['title'] == title].iloc[0]

                    st.markdown(
                        f'<div class="recommendation-card">'
                        f'<strong>#{rank}</strong> Score: {score:.2%}</div>',
                        unsafe_allow_html=True
                    )
                    show_movie_card(movie_info, score)
                    st.divider()

    # ===== TAB 2: Statistics =====
    with tab2:
        st.header(":green[:material/bar_chart:] Dataset Statistics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Movies", len(filtered_df))
        c2.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
        c3.metric("Genres", filtered_df['genre'].nunique())
        c4.metric("Year Range", f"{int(filtered_df['year'].min())}-{int(filtered_df['year'].max())}")

        st.divider()

        # rating distribution
        fig1 = px.histogram(filtered_df, x='rating', nbins=20,
                            title='Rating Distribution',
                            color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig1, use_container_width=True)

        # top genres bar chart
        genre_counts = filtered_df['genre'].value_counts().head(10)
        fig2 = px.bar(x=genre_counts.values, y=genre_counts.index,
                      orientation='h', title='Top 10 Genres',
                      color_discrete_sequence=['#EF553B'])
        fig2.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig2, use_container_width=True)

        # movies per year
        yearly = filtered_df.groupby('year').size().reset_index(name='count')
        fig3 = px.line(yearly, x='year', y='count',
                       title='Movies by Year', markers=True,
                       color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig3, use_container_width=True)

    # ===== TAB 3: Browse =====
    with tab3:
        st.header(":orange[:material/grid_view:] Browse Movies")

        c1, c2, c3 = st.columns(3)
        with c1:
            genre_filter = st.selectbox(
                "Genre:", ["All"] + sorted(filtered_df['genre'].unique().tolist())
            )
        with c2:
            sort_option = st.selectbox(
                "Sort by:", ["Rating (High to Low)", "Rating (Low to High)",
                             "Year (Newest)", "Year (Oldest)"]
            )
        with c3:
            per_page = st.selectbox("Show:", [5, 10, 20, 50])

        browse_df = filtered_df.copy()
        if genre_filter != "All":
            browse_df = browse_df[browse_df['genre'] == genre_filter]

        if sort_option == "Rating (High to Low)":
            browse_df = browse_df.sort_values('rating', ascending=False)
        elif sort_option == "Rating (Low to High)":
            browse_df = browse_df.sort_values('rating', ascending=True)
        elif sort_option == "Year (Newest)":
            browse_df = browse_df.sort_values('year', ascending=False)
        else:
            browse_df = browse_df.sort_values('year', ascending=True)

        for i, (_, movie) in enumerate(browse_df.head(per_page).iterrows(), 1):
            st.markdown(f"### {i}. {movie['title']} ({int(movie['year'])})")
            mc1, mc2, mc3 = st.columns([1, 2, 1])
            mc1.metric("Rating", f"{movie['rating']}/10")
            mc2.write(f"*{movie['genre']}*")
            mc3.metric("Year", int(movie['year']))
            st.divider()

    # ===== TAB 4: About =====
    with tab4:
        st.header(":blue[:material/info:] About This Application")

        st.markdown("""
        ### Problem Statement

        Users face information overload when choosing movies. This application 
        provides personalized movie recommendations using machine learning 
        to analyze movie features and find similar content.

        ### ML Algorithms Used

        **1. Content-Based Filtering**
        - Analyzes movie features (genre, cast, plot keywords)
        - Creates TF-IDF vectors from text features
        - Uses cosine similarity to find closest matches

        **2. Collaborative Filtering**
        - Finds movies with similar genre and rating profiles
        - Uses rating proximity as a similarity measure
        - Approximates user-item patterns with available metadata

        **3. Hybrid Approach**
        - Weighted combination (60% content + 40% collaborative)
        - Merges candidate lists from both methods
        - Normalizes final scores for consistent ranking

        ### Technologies Used

        | Component | Technology |
        |-----------|-----------|
        | Frontend | Streamlit |
        | ML | scikit-learn (TF-IDF, cosine similarity) |
        | Data Processing | Pandas, NumPy |
        | Visualization | Plotly |
        | Architecture | Repository, Service Layer, Strategy Pattern |

        ### Quality Requirements

        1. **Accuracy** — Recommendations should be relevant (measured by 
           cosine similarity scores > 0.3 for content-based matches)
        2. **Performance** — Recommendations generated in < 2 seconds 
           for datasets up to 10,000 movies
        3. **Usability** — Users can get recommendations in 2 clicks or less
        """)


if __name__ == "__main__":
    if 'show_recs' not in st.session_state:
        st.session_state.show_recs = False
    main()
