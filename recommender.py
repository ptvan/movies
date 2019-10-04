import json
from collections import Counter
from keras.models import Model
from keras.layers import Embedding, Input, Reshape
from keras.layers.merge import Dot
from sklearn.linear_model import LinearRegression
import numpy as np
import random
from sklearn import svm

#%%
with open('wp_movies_10k.ndjson') as fin:
    movies = [json.loads(l) for l in fin]

link_counts = Counter()
for movie in movies:
    link_counts.update(movie[2])
link_counts.most_common(10)















#%%
