import pandas as pd
import requests
import re
import time

# read in the OMDBAPI key
txt = pd.read_table("omdbapi.key")
key = list(txt.columns)[0]
base_url = "http://www.omdbapi.com"

# read in our movies
my_movies = pd.read_csv("my_ratings_withtitle.csv")
my_movies["plot"] = ""

# fetch movie synopses
for title in my_movies['title']:
    # print(title)
    params = {'apikey': key, 't': re.sub(" ", "+", title)}
    r = requests.get("http://www.omdbapi.com/", params=params)
    synopsis = r.json().get("Plot")
    my_movies.loc[my_movies['title'] == title, 'plot'] = synopsis
    time.sleep(1)
    # print("\n")

# write back out to new file
my_movies.to_csv("my_ratings_withtitle_withsynopsis.txt", index=False, sep="\t")



