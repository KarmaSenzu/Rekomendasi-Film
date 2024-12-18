import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
import speech_recognition as sr

try:
    df = pd.read_csv('movie_dataset.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['title', 'genres', 'director', 'cast', 'keywords', 'user_id', 'vote_average'])

df.fillna('', inplace=True)

def recommend_movies(user_input):
    vectorizer = TfidfVectorizer(stop_words='english')
    df['combined_features'] = df['title'] + ' ' + df['genres'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['keywords']
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    user_input = user_input.lower()
    user_input_vector = vectorizer.transform([user_input])
    
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-6:][::-1]
    
    recommended_movies = df.iloc[similar_indices]
    return recommended_movies[['title', 'genres', 'director', 'cast', 'keywords', 'vote_average']]

def get_movies_by_keyword(user_input):
    return df[df['keywords'].str.contains(user_input, case=False)][['title', 'genres', 'director', 'cast', 'keywords', 'vote_average']]

def get_movies_by_genre(user_input):
    genres = df['genres'].unique()
    closest_match = difflib.get_close_matches(user_input, genres, n=1, cutoff=0.6)
    if closest_match:
        genre_match = closest_match[0]
        return df[df['genres'].str.contains(genre_match, case=False)][['title', 'genres', 'director', 'cast', 'keywords', 'vote_average']]

def get_movies_by_director(user_input):
    directors = df['director'].unique()
    closest_match = difflib.get_close_matches(user_input, directors, n=1, cutoff=0.6)
    if closest_match:
        director_match = closest_match[0]
        return df[df['director'].str.contains(director_match, case=False)][['title', 'genres', 'director', 'cast', 'keywords', 'vote_average']]

def get_movies_by_cast(user_input):
    cast = df['cast'].unique()
    closest_match = difflib.get_close_matches(user_input, cast, n=1, cutoff=0.6)
    if closest_match:
        cast_match = closest_match[0]

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🔊 Silakan berbicara sekarang...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="id-ID")
            st.write(f"Anda berkata: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Google Speech Recognition tidak bisa memahami audio.")
            return ""
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

st.set_page_config(page_title="Pencarian dan Rekomendasi Film", layout="wide")

st.title("🎥 Aplikasi Pencarian dan Rekomendasi Film")

option = st.sidebar.selectbox(
    "Pilih jenis pencarian:",
    ["Cari Berdasarkan Kata Kunci", "Cari Berdasarkan Genre", "Cari Berdasarkan Sutradara", "Rekomendasi Film"]
)

def display_results(data, message="Hasil pencarian Anda:"):
    if not data.empty:
        data.columns = ["Judul", "Genre", "Sutradara", "Pemeran", "Kata Kunci", "Rating"]

        data = data.sort_values(by="Rating", ascending=False)
        
        st.write(f"### {message}")

        st.dataframe(data.style.format({"Rating": "{:.1f}"}))

if option == "Cari Berdasarkan Kata Kunci":
    keyword = st.text_input("Masukkan kata kunci:")
    if st.button("Cari"):
        result = get_movies_by_keyword(keyword)
        display_results(result)
    
    if st.button("Gunakan Suara untuk Mencari"):
        keyword = speech_to_text()
        if keyword:
            result = get_movies_by_keyword(keyword)
            display_results(result)

elif option == "Cari Berdasarkan Genre":
    genre = st.text_input("Masukkan genre:")
    if st.button("Cari"):
        result = get_movies_by_genre(genre)
        display_results(result)
    
    if st.button("Gunakan Suara untuk Mencari"):
        genre = speech_to_text()
        if genre:
            result = get_movies_by_genre(genre)
            display_results(result)

elif option == "Cari Berdasarkan Sutradara":
    director = st.text_input("Masukkan nama sutradara:")
    if st.button("Cari"):
        result = get_movies_by_director(director)
        display_results(result)
    
    if st.button("Gunakan Suara untuk Mencari"):
        director = speech_to_text()
        if director:
            result = get_movies_by_director(director)
            display_results(result)

elif option == "Cari Berdasarkan Pemeran":
    cast = st.text_input("Masukkan nama pemeran:")
    if st.button("Cari"):
        result = get_movies_by_cast(cast)
        display_results(result)
    
    if st.button("Gunakan Suara untuk Mencari"):
        cast = speech_to_text()
        if cast:
            result = get_movies_by_cast(cast)
            display_results(result)

elif option == "Rekomendasi Film":
    user_input = st.text_input("Masukkan detail film atau preferensi:")
    if st.button("Rekomendasi"):
        result = recommend_movies(user_input)
        display_results(result, message="Film yang direkomendasikan untuk Anda:")
    
    if st.button("Gunakan Suara untuk Rekomendasi"):
        user_input = speech_to_text()
        if user_input:
            result = recommend_movies(user_input)
            display_results(result, message="Film yang direkomendasikan untuk Anda:")
