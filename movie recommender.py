import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'title': [
        'Avatar', 'Titanic', 'The Avengers', 'Iron Man',
        'The Dark Knight', 'Inception', 'Interstellar'
    ],
    'genre': [
        'Action Adventure Sci-Fi',
        'Romance Drama',
        'Action Superhero',
        'Action Superhero',
        'Action Crime Drama',
        'Sci-Fi Thriller',
        'Sci-Fi Drama'
    ]
}

df = pd.DataFrame(data)

cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

similarity = cosine_similarity(matrix)

def recommend(movie):
    if movie not in df['title'].values:
        return "Movie not found!"

    index = df[df['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:4]

    for i in movies:
        print(df.iloc[i[0]]['title'])

recommend('Avatar')