library(sparklyr)
# spark_install(version = "2.4")

## NOTE: Spark 2.x only works on Java 8, so the below assume an OSX environment and
## that we have run
## brew cask install homebrew/cask-versions/adoptopenjdk8 
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home")

conf <- list()

conf$`sparklyr.cores.local` <- 4
conf$`sparklyr.shell.driver-memory` <- "8G"
conf$spark.memory.fraction <- 0.6

sc <- spark_connect(master = "local", 
                    version = "2.4",
                    config = conf)

# read data directly from disk
movies <- spark_read_csv(sc, "dat","movies.csv")

# alternatively, we can copy objects from the R session into the Spark scope
movies <- copy_to(sc, movies, "movies")

# list contents of our spark context
src_tbls(sc)

# build an ALS model for our movies
model <- ml_als(movies)

# predict
ml_predict(model, movies_tbl)

# recommend
ml_recommend(model, type = "item", 1)