import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("../data/movies.csv")
ratings = pd.read_csv("../data/ratings.csv")

# -------- SVD MODEL --------
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

trainset = data.build_full_trainset()
svd_model = SVD()
svd_model.fit(trainset)

# -------- TF-IDF MODEL --------
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save everything
joblib.dump({
    "svd": svd_model,
    "movies": movies,
    "cosine_sim": cosine_sim
}, "saved_model.pkl")

print("✅ Model trained and saved!")