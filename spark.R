library(sparklyr)
library(dplyr)
library(data.table)
# spark_install(version = "2.4")

## NOTE: Spark 2.x only works on Java 8, so the below assume an OSX environment and
## that we have run
## brew cask install homebrew/cask-versions/adoptopenjdk8 
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home")

conf <- list()

conf$`sparklyr.cores.local` <- 6
conf$`sparklyr.shell.driver-memory` <- "8G"
conf$spark.memory.fraction <- 0.6

sc <- spark_connect(master = "local", 
                    version = "2.4",
                    config = conf)

# read data into R from disk
ratings <- fread("movielens-20m/ratings.csv")
my_ratings <- fread("my_ratings.csv")

# copy to Spark context
my_ratings <- copy_to(sc, my_ratings)
ratings <- copy_to(sc, ratings)

# list contents of our spark context
src_tbls(sc)

# build an ALS model for our movies
model <- ml_als(ratings, rating ~ userId + movieId)

# predict
ml_predict(model, movies_tbl)

# recommend
ml_recommend(model, type = "movieId", 1)