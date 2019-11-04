import json
from collections import Counter
from keras.models import Model
from keras.layers import Embedding, Input, Reshape
from keras.layers.merge import Dot
from sklearn.linear_model import LinearRegression
import numpy as np
import random
from sklearn import svm

# from Deep Learning Cookbook
#%%
with open('wp_movies_10k.ndjson') as fin:
    movies = [json.loads(l) for l in fin]

link_counts = Counter()
for movie in movies:
    link_counts.update(movie[2])

top_links = [link for link, c in link_counts.items() if c >= 3]
link_to_idx = {link: idx for idx, link in enumerate(top_links)}
movie_to_idx = {movie[0]: idx for idx, movie in enumerate(movies)}
pairs = []
for movie in movies:
    pairs.extend((link_to_idx[link], movie_to_idx[movie[0]]) for link in movie[2] if link in link_to_idx)
pairs_set = set(pairs)
len(pairs), len(top_links), len(movie_to_idx)














#%%
