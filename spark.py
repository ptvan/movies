import pyspark
from pyspark import SparkContext
sc = SparkContext("local", "Movie Recommender")
my_ratings = sc.textFile("my_ratings.csv")
all_ratings = sc.textFile("movielens-20m/ratings.csv")
