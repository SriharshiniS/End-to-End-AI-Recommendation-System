from flask import Flask, render_template, request
from model.recommender import hybrid_recommendation

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        user_id = int(request.form["user_id"])
        movie = request.form["movie"]

        recommendations = hybrid_recommendation(user_id, movie)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)