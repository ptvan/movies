import pandas as pd
import seaborn as sb

# read in the original CSV file which has the format
# {title, my_rating}
my_ratings = pd.read_csv('my_ratings.csv')

# getting a glimpse of the data
my_ratings.count()
list(my_ratings.columns)
my_ratings.head()

