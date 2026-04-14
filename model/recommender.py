import joblib
import pandas as pd

data = joblib.load("model/saved_model.pkl")

svd = data["svd"]
movies = data["movies"]
cosine_sim = data["cosine_sim"]

# Reverse index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_recommendations(title, top_n=5):
    idx = indices.get(title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()


def get_collab_recommendations(user_id, top_n=5):
    movie_ids = movies['movieId'].unique()
    predictions = []

    for movie_id in movie_ids:
        pred = svd.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = [x[0] for x in predictions[:top_n]]
    return movies[movies['movieId'].isin(top_movies)]['title'].tolist()


def hybrid_recommendation(user_id, title):
    content = get_content_recommendations(title)
    collab = get_collab_recommendations(user_id)

    return list(set(content + collab))