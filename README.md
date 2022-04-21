# Recomendation system using Content-based approach

## Introduction

This is a realization of a content-based recomandation system for proposing games to users based on their previos ratings. I chose this kind of system specificly because it doesn't require to keep in mind the past dependencies, time stamps and the complex analys of behavior, we just predict basing on the 2 facts. For the main tasks I used ordinary ML libraries as `numpy`, `pandas`, `sklearn` and others. For releasing API attempt I used `Flask` framework. 

###In addition

1. All data are stoed in `data` folder in the `root`; html-pages for `API services` is in `templates` folder.

2. Here lies `requirements.txt` with all add-ons and `ml_task.ipynb` - Goggle Collaboratory notebook just to see how algorithm works step-by-step.

Google Colaboratory is available in this link too for the simplicity: https://colab.research.google.com/drive/1qkSfg0ScxR6MOMNmW961zjjIc2RQyqIT#scrollTo=ezCbJiCgfuh1 

## Installation

Firstly you need to install all the necessary libraries:

```bash
pip install numpy
pip install scipy
pip install pandas
pip install sklearn
pip install nltk
```

Then download the script from the repository:

```bash
git clone https://github.com/sashaismonster/ml-task-internship.git
```

## Code explanation

Import libraries:

```python
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
```
Read games and users data using `read_csv()` methods and remame some colums to follow PEP 8 conventions.

Then we need to check the data for 100% validness:
```python
def clean_reatings(rating):
  if rating and type(rating) == str:
    rating = rating.replace(';', '')
    if rating.count('.') > 1 and rating[-1] == '.':
      return rating[:-1]
    elif float(rating) > 10:
      return np.nan
    else:
      return rating
  return rating
```
Replace NuLL values and convert it into the appropriate for following manipulation form (1000x250 matrix):

```python
df_users_ratings['rating'].fillna(df_users_ratings['rating'].mean(), inplace=True)
df_game_features = df_users_ratings.pivot(index='userId', columns='gameId', values='rating').fillna(0)
```

The main thing we need to proceed for the content-based algorithm execution is SVD decomposition:

```python
R = df_game_features.values
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned)

sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_game_features.columns)
```

And finally recieve a result in this function, it lieas in `recomandations` variable:

```python
def recommend_games(preds_df, userId, games_df, original_ratings_df, num_recommendations=5): #default we print top 5 recomndations
  user_row_number = userId - 1 
  sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
  user_data = original_ratings_df[original_ratings_df.userId == (userId)]
  user_full = (user_data.merge(games_df, how = 'left', left_on = 'gameId', right_on = 'gameId').
                    sort_values(['rating'], ascending=False)
                )
  recommendations = (games_df[~games_df['gameId'].isin(user_full['gameId'])]).merge(
      pd.DataFrame(sorted_user_predictions).reset_index(),
      how = 'left', left_on = 'gameId', right_on = 'gameId'
      ).rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
                    

  return user_full, recommendations
```

## Interaction

In the consol you are asked to fill in `userID` of a user for whom to predict and ampunt of recomandation
The answer lies in `DataFrame` that is printed on the screen and conversted to dictionary to parse it in API as `JSON` file.

## API service

Currently doesn't work, it's need to be updated
I used `flask` library need to be installed to with the following dependencies:

```python
import flask
from flask import request, jsonify
from flask import Flask, render_template
from main import to_show
```

It containes the form in `index.html`, submit data via `POST/GET` protocols and then parses the answer using `jsonify()` embedded functions. May be useful for some purposes

