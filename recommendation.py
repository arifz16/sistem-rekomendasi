"""
File: recommendation.py
Description: Modul logika rekomendasi yang mengembalikan metadata film.
"""

import pickle
import pandas as pd
import difflib

class MovieRecommender:
    def __init__(self, movies_pkl_path='movies.pkl', similarity_pkl_path='similarity.pkl'):
        self.movies_df = self.load_model(movies_pkl_path)
        self.similarity_matrix = self._load_pickle(similarity_pkl_path)
        self.all_titles = self.movies_df['title'].str.lower().tolist()
        
    def _load_pickle(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
            
    def load_model(self, path):
        movie_dict = self._load_pickle(path)
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
            
            # Ambil detail lengkap dari dataframe
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