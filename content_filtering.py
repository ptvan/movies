import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import linear_kernel
from rake_nltk import Rake
# import seaborn as sb

# read in the original CSV file which has the format
# {title, my_rating}
my_ratings = pd.read_csv('my_ratings_withtitle_withsynopsis.txt', sep="\t", encoding="ISO-8859-1")

# getting a glimpse of the data
my_ratings.count()
list(my_ratings.columns)
my_ratings.head()

# read in public user tags from MovieLens
movielens_tags = pd.read_csv('movielens-20m/tags.csv')

# generate TF-IDF matrix
data = my_ratings

data['keywords'] = ""

for i in range(data.shape[0]):
    plot = data.iloc[i]['plot']
    # print(plot)
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whit key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()

    # assigning the key words to the new column for the corresponding movie
    data.at[i, 'keywords'] = ' '.join(map(str, list(key_words_dict_scores.keys())))


count = CountVectorizer()
count_matrix = count.fit_transform(data['Key_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# create an index for movie titles
indices = pd.Series(data.title)

# function that does the actual matching
def recommendations(title, cosine_sim = cosine_sim):
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
        recommended_movies.append(list(data.index)[i])

    return recommended_movies

# map the indices back to movie titles
indices[recommendations("The Kid")]

# merge with my_ratings
# data = pd.merge(my_ratings, movielens_tags, on="movieId", how="inner")
# data = data.drop(["user_id", "userId", "timestamp"], axis=1)

# collapse the tags for each movie into a single row
# data = data.groupby(['movieId', 'rating'])['tag'].apply(' '.join).reset_index()
