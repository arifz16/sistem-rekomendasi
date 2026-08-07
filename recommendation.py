"""
File: recommendation.py
Description: Menghitung similarity secara on-the-fly untuk menghindari file .pkl berukuran besar.
"""

import pickle
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_pkl_path='movies.pkl'):
        # 1. Load data film (hanya butuh file movies.pkl)
        self.movies_df = self.load_model(movies_pkl_path)
        self.all_titles = self.movies_df['title'].str.lower().tolist()
        
        # 2. Hitung Kemiripan (Similarity) secara On-The-Fly
        # Membaca kolom 'tags' lalu membuat matriks similarity secara instan
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.8)
        tfidf_matrix = tfidf.fit_transform(self.movies_df['tags']).toarray()
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
    def load_model(self, path):
        with open(path, 'rb') as file:
            movie_dict = pickle.load(file)
        return pd.DataFrame(movie_dict)

    def search_movie(self, movie_title):
        movie_title_lower = movie_title.lower()
        close_match = difflib.get_close_matches(movie_title_lower, self.all_titles, n=1, cutoff=0.6)
        
        if close_match:
            return close_match[0]
        return None

    def recommend(self, movie_title, top_n=10):
        match_title = self.search_movie(movie_title)
        
        if not match_title:
            return pd.DataFrame() 
            
        movie_index = self.all_titles.index(match_title)
        distances = self.similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        
        results = []
        for i in movies_list:
            idx = i[0]
            score = i[1]
            
            title = self.movies_df.iloc[idx]['title']
            genres = self.movies_df.iloc[idx]['genres_display']
            overview = self.movies_df.iloc[idx]['overview_display']
            
            results.append({
                'title': title,
                'similarity_score': round(score, 4),
                'genres': genres,
                'overview': overview
            })
            
        return pd.DataFrame(results)
