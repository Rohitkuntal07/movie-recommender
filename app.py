import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['genres'] = movies['genres'].fillna('')

# Build model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
similarity = cosine_similarity(tfidf_matrix)

def recommend(title):
    title = title.lower()
    if title not in movies['title'].str.lower().values:
        return ["Movie not found."]
    
    index = movies[movies['title'].str.lower() == title].index[0]
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[recommended_indices].tolist()

# Streamlit interface
st.title("🎬 Movie Recommender")

movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)
