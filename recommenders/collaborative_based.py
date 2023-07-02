#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ignore warnings
import warnings
warnings.simplefilter(action='ignore')


# In[2]:


# Import our regular old heroes 
import numpy as np
import pandas as pd
import scipy as sp # <-- The sister of Numpy, used in our code for numerical efficientcy. 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

import surprise
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise import SVDpp
from surprise import accuracy

from surprise.model_selection import train_test_split

# Entity featurization and similarity computation
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Libraries used during sorting procedures.
import operator # <-- Convienient item retrieval during iteration 
import heapq # <-- Efficient sorting of large lists

from surprise import NMF
from surprise import Dataset
from surprise import accuracy
from surprise import BaselineOnly
from surprise import accuracy


# In[3]:


# Imports all movie data

df_genome_scores= pd.read_csv('genome_scores.csv')
df_genome_tags= pd.read_csv('genome_tags.csv')
df_imdb_data= pd.read_csv('imdb_data.csv')
df_links= pd.read_csv('links.csv')
df_movies= pd.read_csv('movies.csv')
df_tags= pd.read_csv('tags.csv')
df_train= pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[4]:


df_train.head()


# In[44]:


#Looking at all the data
df_genome_scores.info()


# In[45]:


df_genome_scores.head()


# In[7]:


df_genome_tags.info()


# In[8]:


df_genome_tags.head()


# In[9]:


df_tags.info()


# In[10]:


df_tags.head(50)


# In[11]:


#df_tags['tag'] = df_tags['tag'].str.replace(' ', '').str.strip()
df_tags['tag'] = df_tags['tag'].str.replace('|', ' ').str.strip()


# In[12]:


#Droping userId and timestam from tags
df_tags = df_tags.drop(['userId', 'timestamp'], axis=1)
df_tags


# In[13]:


# Group the `top` dataframe by `movieId` and join the `tag` values in the `tag` column separated by a comma
df_tags.reset_index(drop=True, inplace=True)
#Convert the tag to a string value
df_tags['tag'] = df_tags['tag'].astype(str)
group_tags = df_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
group_tags


# In[14]:


df_movies.head()


# In[15]:


#change the genres coloumn into a list
df_movies['genres'] = df_movies['genres'].str.replace('|', ' ').str.strip()
df_movies.head()


# In[16]:


df_movies.info()


# In[17]:


#Remove duplicate Movies if any
df_movies = df_movies.drop_duplicates(subset='movieId', keep='first')


# In[18]:


#Link movies with tags
df_movies_train = pd.merge(df_train, df_movies, on='movieId', how='left')


# In[19]:


df_movies_train


# In[20]:


df_imdb_data.head()


# In[21]:


#Fill all the null values and reolacing the space with a no space for the names
df_imdb_data['title_cast'] = df_imdb_data['title_cast'].fillna('')
#df_imdb_data['title_cast'] = df_imdb_data['title_cast'].str.replace(' ', '').str.strip()
df_imdb_data['title_cast'] = df_imdb_data['title_cast'].str.replace('|', ' ').str.strip()

df_imdb_data['plot_keywords'] = df_imdb_data['plot_keywords'].fillna('')
df_imdb_data['plot_keywords'] = df_imdb_data['plot_keywords'].str.replace('|', ' ').str.strip()

df_imdb_data['director'] = df_imdb_data['director'].fillna('')
#df_imdb_data['director'] = df_imdb_data['director'].str.replace(' ', '').str.strip()
df_imdb_data['director'] = df_imdb_data['director'].str.replace('|', ' ').str.strip()

df_imdb_data['runtime'] = df_imdb_data['runtime'].fillna(0)
df_imdb_data['budget'] = df_imdb_data['budget'].fillna('')

df_imdb_data.head()


# In[22]:


#Link movies with imdb Data
df_movies_train = pd.merge(df_movies_train, df_imdb_data, on='movieId', how='left')


# In[23]:


#Taking out the year of the tite into a new coloumn
df_movies_train['year'] = df_movies_train.title.str.extract('(\(\d\d\d\d\))', expand=False)
df_movies_train['year'] = df_movies_train.year.str.extract('(\d\d\d\d)',expand=False)
df_movies_train['title'] = df_movies_train.title.str.replace('(\(\d\d\d\d\))','')

df_movies_train['title']=df_movies_train['title'].apply(lambda x: x.strip())


# In[24]:


df_movies_train


# In[25]:


#Drop the timestamp
df_movies_train = df_movies_train.drop('timestamp', axis=1)


# In[26]:


df_movies_train.info()


# In[27]:


#Make a graph of all the ratings
with sns.axes_style('white'):
    g = sns.catplot(x="rating", data=df_movies_train, aspect=2.0, kind='count')
    g.set_ylabels("Total number of ratings")

average_rating = np.mean(df_movies_train["rating"])
print(f'Average rating in dataset: {average_rating}')


# In[28]:


# Have a look at all the unique users and movies in the dataset.
print('Unique users in training dataset: ', df_train['userId'].nunique())
print('Unique movies in training dataset: ', df_train['movieId'].nunique())


# In[29]:


#drop timestamp from axis
df_train = df_train.drop('timestamp', axis=1)
df_train.info()


# In[30]:


#Count ratings per user
rating_counts = df_train['userId'].value_counts()
print(rating_counts)


# In[31]:


#User 72315 is an outlier with to many ratings
#remove users 72315
df_train = df_train[df_train['userId'] != 72315]

rating_counts = df_train['userId'].value_counts()
print(rating_counts)


# In[33]:


#train the data set on the full df_train data
reader = Reader(rating_scale=(min(df_train['rating']), max(df_train['rating'])))
df_train_subset = df_train[['userId', 'movieId', 'rating']]
df_train = Dataset.load_from_df(df_train_subset, reader=reader)

SVDpp_hyper_final_model_train = df_train.build_full_trainset()

#####SVDpp_hyper = SVDpp()
#####SVDpp_hyper.fit(SVDpp_hyper_final_model_train)


# In[34]:


# Split the data into training and test sets
trainset, testset = train_test_split(df_train, test_size=0.2, random_state=42)


# In[ ]:


# Split the data into training and test sets
trainset, testset = train_test_split(df_train, test_size=0.2)

# Train the BaselineOnly model
baseline_algo = BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 60, 'reg': 0.05, 'learning_rate': 0.01})
baseline_algo.fit(trainset)

# Make predictions on the test set
predictions = baseline_algo.test(testset)

# Compute RMSE
rmse = accuracy.rmse(predictions)

print("The RMSE on df_train: ", rmse)

# Improve the BaselineOnly model with SGD optimization
improved_baseline_algo = BaselineOnly(bsl_options={'method': 'sgd', 'n_epochs': 60, 'reg': 0.05, 'learning_rate': 0.01})
improved_baseline_algo.fit(trainset)

# Make predictions on the test set using the improved model
improved_predictions = improved_baseline_algo.test(testset)

# Compute RMSE with the improved model
improved_rmse = accuracy.rmse(improved_predictions)

print("The improved RMSE on the test set: ", improved_rmse)


# In[ ]:


# Train the NMF model
nmf_model = NMF()
nmf_model.fit(trainset)

# Make predictions on the test set
predictions = nmf_model.test(testset)

# Compute RMSE
rmse = accuracy.rmse(predictions)

print("The RMSE on the test set: ", rmse)


# In[46]:


# Train the SVDpp model 1
SVDpp_hyper = SVDpp(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
SVDpp_hyper.fit(trainset)

# Make predictions on the test set
predictions = SVDpp_hyper.test(testset)

# Compute RMSE
rmse = accuracy.rmse(predictions)


print("The rmse on df_train: ", rmse)


# In[35]:


#Test on different values
# Train the SVDpp model 2
SVDpp_hyper = SVDpp(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
SVDpp_hyper.fit(trainset)

# Make predictions on the test set
predictions = SVDpp_hyper.test(testset)
mmnm,("")
# Compute RMSE
rmse = accuracy.rmse(predictions)


print("The rmse on df_train: ", rmse)


# In[43]:


#Test on different values
# Train the SVDpp model 3
SVDpp_hyper = SVDpp(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.03)
SVDpp_hyper.fit(trainset)

# Make predictions on the test set
predictions = SVDpp_hyper.test(testset)

# Compute RMSE
rmse = accuracy.rmse(predictions)


print("The rmse on df_train: ", rmse)


# In[47]:


#Predict the Ratings for the test data set
predictions = []
for index, row in df_test.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    prediction = SVDpp_hyper.predict(user_id, movie_id)
    predictions.append(prediction)


# In[48]:


#Check if all the pridictions are there
num_predictions = len(predictions)
print(f"Number of predictions: {num_predictions}")
predictions_df = pd.DataFrame(predictions)


# In[49]:


#Displal prediction
first_prediction = predictions[3]
print(first_prediction)


# In[50]:


# Convert predictions to a list of dictionaries
predictions_dict = [{'uid': pred.uid, 'iid': pred.iid, 'rating': pred.est} for pred in predictions]

# Create a DataFrame from the predictions
predictions_df = pd.DataFrame(predictions_dict)

# Combine uid and iid into uid_iid column
predictions_df['uid_iid'] = predictions_df['uid'].astype(str) + '_' + predictions_df['iid'].astype(str)

# Select only the 'uid_iid' and 'est' columns
predictions_df = predictions_df[['uid_iid', 'rating']]
predictions_df.rename(columns={'uid_iid': 'id'}, inplace=True)
# Print the first few rows of the reshaped DataFrame
predictions_df.head()


# In[51]:


#Create Csv file
predictions_df.to_csv('submission.csv', index=False)


# In[52]:


#Here is the github link

#https://github.com/RiedelJ/EDSA_Movie.git


# In[46]:


#First of I tried a Content based filtering as there was a lot of data to support my idea.
#First I grouped the tag data together by the movieId to get all the tags together. Then I droped the 'userId', 'timestamp'
#from the tags data.
#I Then I replaced the movies gernes | with a space that I get all the keywords and droped all the duplicats.
#I then merged the training data with the movie data and droped the 'timestamp'
#Then in the imdb_data I replaced the | with a space in the following coloumns title_cast,director and plot_keywords
#After that I merged it with the movie data that haaf been merged with th train data already and extracted the year from the title 
#into another field. Then I droped the 'timestamp'
#Then I displayed all the rating to make have a look how the users are rating the movies.
#After trying to process the data, My laptop allways ran out of Memory and I had to find another solution. 
#After building a Server with 64GB Ram it finaly executed and after 50hours of processing it gave me a score of 1.1 on Kaggle.

#After that I decided to use Colleborative filtering.
#In the training data there is more than enough data to use  Colleborative filtering.
#First I checked how many numique users and movies I have.
#Then I removed the timestamp from the training data.
#After that I checked for outliers and saw there is an outlier. The first user with over 12000 ratings. that is almost impossible so I
#removed him. 
#Then I did a train test split
#
#BaselineOnly
#The BaselineOnly did not prefore that well and was alot higher than my there algorithems
#
#NMF
#The NMF was also much higher than the SVDpp
#
#SVDpp
# I used the SVDpp to do the split as I got the best results. I tryed to do a GridSearchCV, but it took too long so I tried to manually 
# set the hypermeters. These where the best values I got. SVDpp(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.03)
#Then I ran it on my test data and exported it to an CSV


# In[ ]:




