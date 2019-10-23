import pyspark
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math

# remember to add `export PYSPARK_PYTHON=/usr/bin/python3` to ~/.bash_profile
# or Python version mismatch will occur

sc = SparkContext("local", "Movie Recommender")
my_ratings = sc.textFile("my_ratings.csv")
all_ratings = sc.textFile("movielens-20m/ratings.csv")

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1

training, test = all_ratings.randomSplit([7, 3], seed=0)

model = ALS.train(training, best_rank, seed=seed,
                  iterations=iterations, lambda_=regularization_parameter)
