import pandas as pd
import requests

# read in the OMDBAPI key
txt = pd.read_table("omdbapi.key")
key = list(txt.columns)[0]

# fetch movie metadata
params = {'apikey':key}
r = requests.get("http://www.omdbapi.com/?i=tt3896198", params=params)
synopsis = r.json().get("Plot")

