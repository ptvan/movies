import pyspark
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import os
import math

# remember to add `export PYSPARK_PYTHON=/path/to/python3/` to ~/.bash_profile
# or Python version mismatch will occur

# if installing spark on OSX, make sure to install adoptopenjdk8
# and `export JAVA_HOME=/path/to/openjdk8`

sc = SparkContext("local", "Movie Recommender")
my_ratings = sc.textFile("my_ratings.csv")
small_ratings_raw = sc.textFile(os.path.join('movielens-100k', 'ratings.csv'))
small_ratings_raw_header = small_ratings_raw.take(1)[0]
small_ratings = small_ratings_raw.filter(lambda line: line != small_ratings_raw_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()


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

training_RDD, validation_RDD, test_RDD = small_ratings.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
   # print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

# print 'The best model was trained with rank %s' % best_rank

predictions.take(3)

model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                  lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())


complete_ratings_raw_data = sc.textFile(os.path.join('movielens-20m', 'ratings.csv'))
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0)

complete_model = ALS.train(training_RDD, best_rank, seed=seed,
                           iterations=iterations, lambda_=regularization_parameter)


my_ID = 0

### LOAD my ratings
# The format of each line is (userID, movieID, rating)
my_ratings = [
     (0,1,5), # Toy Story (1995)
     (0,296,3), # Pulp Fiction (1994)
     ]

my_ratings_RDD = sc.parallelize(my_ratings)
complete_data_with_my_ratings_RDD = complete_ratings_data.union(my_ratings_RDD)
new_ratings_model = ALS.train(complete_data_with_my_ratings_RDD, best_rank, seed=seed,
                              iterations=iterations, lambda_=regularization_parameter)