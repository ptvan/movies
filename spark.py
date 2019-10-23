import pyspark
from pyspark import SparkContext
sc = SparkContext("local", "Movie Recommender")
sc.addFile("my_ratings.csv")
