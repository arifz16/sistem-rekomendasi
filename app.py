"""
File: app.py
Description: Antarmuka Web Streamlit (Offline / Standalone Dataset)
"""

import streamlit as st
from recommendation import MovieRecommender

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_resource
def load_recommender():
    return MovieRecommender()

# Load model
recommender = load_recommender()
movies_list = recommender.movies_df['title'].values

# Sidebar
with st.sidebar:
    st.title("🎬 AI Recommender")
    st.write("Versi: Standalone (Tanpa API)")
    st.markdown("""
    **Fitur yang digunakan:**  
    ✅ Title  
    ✅ Overview  
    ✅ Genres  
    ✅ Keywords  
    
    *Tidak membutuhkan koneksi internet (TMDB API) untuk menampilkan data.*
    """)

# Main Content
st.title("Sistem Rekomendasi Film Berbasis Konten")
st.write("Temukan film yang mirip berdasarkan jalan cerita dan genre.")

selected_movie = st.selectbox("Cari atau pilih film preferensi Anda:", movies_list)

if st.button("Rekomendasikan Film!", type='primary'):
    with st.spinner("Mencari film dengan plot serupa..."):
        recommendations = recommender.recommend(selected_movie, top_n=10)
        
    if recommendations.empty:
        st.error("Film tidak ditemukan di database.")
    else:
        st.success(f"Ditemukan 10 film yang mirip dengan **{selected_movie}**:")
        
        # Menampilkan hasil dalam bentuk daftar vertikal yang elegan
        for idx, row in recommendations.iterrows():
            with st.container():
                st.subheader(f"{idx + 1}. {row['title']}")
                st.caption(f"⭐ **Similarity Score:** {row['similarity_score']} | 🎭 **Genres:** {row['genres']}")
                st.write(f"**Sinopsis:** {row['overview']}")
                st.divider() # Garis pemisah antar film