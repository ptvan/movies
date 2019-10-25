import pandas as pd

# read in the original CSV file which has the format
# {title, my_rating}
my_ratings = pd.read_csv('my_ratings.csv')

# getting a glimpse of the data
my_ratings.count()
list(my_ratings.columns)
my_ratings.head()

# filter out movies I haven't watched (ie. ratings == 0)
my_ratings = my_ratings[my_ratings["rating"] != 0]


# running ALS requires the data to have the format
# {userid, movieid, rating}
# create a new column with bogus userID 0
my_ratings.insert(0, "UserID", 0)

# renaming columns
my_ratings.rename(columns={'my_rating': 'rating'}, inplace=True)
my_ratings.rename(columns={'UserID': 'user_id', 'title': 'title'}, inplace=True)

# read in the MovieLens movies
all_movies = pd.read_csv('movielens-20m/movies.csv')

# we don't need the genres
all_movies = all_movies.drop(columns=['genres'])

# remove the year of release from titles
all_movies["title"] = all_movies["title"].replace(" \\([0-9]{4}\\)", "", regex=True)

# merge with my_ratings
merged_ratings = pd.merge(all_movies, my_ratings, on='title', how='inner')

# some movies have same titles, eg. very short titles or Shakespeare adaptations
merged_ratings.count() # 281 movies
len(merged_ratings.title.unique())  # 219 unique titles
merged_ratings[merged_ratings.duplicated(subset="title")]
merged_ratings[merged_ratings['title'] == "Beauty and the Beast"]



