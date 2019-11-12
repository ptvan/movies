import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# import seaborn as sb

# read in the original CSV file which has the format
# {title, my_rating}
my_ratings = pd.read_csv('my_ratings_withtitle_withsynopsis.csv')

# getting a glimpse of the data
my_ratings.count()
list(my_ratings.columns)
my_ratings.head()

# read in public user tags from MovieLens
movielens_tags = pd.read_csv('movielens-20m/tags.csv')

# merge with my_ratings
data = pd.merge(my_ratings, movielens_tags, on="movieId", how="inner")
data = data.drop(["user_id", "userId", "timestamp"], axis=1)

# collapse the tags for each movie into a single row
data = data.groupby(['movieId', 'rating'])['tag'].apply(' '.join).reset_index()

# generate TF-IDF matrix
v = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = v.fit_transform(data['tag'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
indices = pd.Series(data.index)


#  defining the function that takes in movie title
# as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim=cosine_sim):
    # initializing the empty list of recommended movies
    recommended_movies = []

    # getting the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    return recommended_movies