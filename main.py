# Import libraries and dependencies
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

#Clean-up function, where the incorrect fields are changed, and invalid values are thrown out
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

#Content-based recomendation function
def recommend_games(preds_df, userId, games_df, original_ratings_df,
                    num_recommendations=5):  # default we print top 5 recomndations
    user_row_number = userId - 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    user_data = original_ratings_df[original_ratings_df.userId == (userId)]
    user_full = (user_data.merge(games_df, how='left', left_on='gameId', right_on='gameId').
                 sort_values(['rating'], ascending=False)
                 )
    recommendations = (games_df[~games_df['gameId'].isin(user_full['gameId'])]).merge(
        pd.DataFrame(sorted_user_predictions).reset_index(),
        how='left', left_on='gameId', right_on='gameId'
    ).rename(columns={user_row_number: 'Predictions'}).sort_values('Predictions', ascending=False).iloc[
                      :num_recommendations, :-1]

    return user_full, recommendations



# Read games and users data
df_games = pd.read_csv('./data/games.csv', low_memory=False)
df_users_ratings = pd.read_csv('./data/user.csv', low_memory=False)

# Remame some columns to follow PEP 8 convention
df_games.rename(columns={'id_game': 'gameId'}, inplace=True)
df_users_ratings.rename(columns={'rating;': 'rating', 'UserID': 'userId', 'GameID': 'gameId'}, inplace=True)

# Clean up rating data to ensure that all data is valid
df_users_ratings['rating'] = df_users_ratings['rating'].apply(lambda x: clean_reatings(x)).apply(pd.to_numeric)
df_users_ratings.head()

# Replace null values
df_users_ratings['rating'].fillna(df_users_ratings['rating'].mean(), inplace=True)

#making tables of pivots in order to represent it in clear manner to put in SVD decomposition and apply content based system
df_game_features = df_users_ratings.pivot(index='userId', columns='gameId', values='rating').fillna(0)

#SVD decomposition
R = df_game_features.values
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned)

sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_game_features.columns)

print("type userID for next recomendation:")
userID = int(input())  # for whom we predict
print("type amount of recomendations needed for him:")
recNumber = int(input()) # how much predictions to show

#MAIN PART
def to_show():
    already_rated, predictions = recommend_games(preds_df, userID, df_games, df_users_ratings, recNumber)
    return predictions.head(recNumber)

print(to_show())