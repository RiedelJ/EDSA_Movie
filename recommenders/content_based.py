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

# Entity featurization and similarity computation
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer

# Libraries used during sorting procedures.
import operator # <-- Convienient item retrieval during iteration 
import heapq # <-- Efficient sorting of large lists


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


df_movies.head()


# In[5]:


#change the genres coloumn into a list
df_movies['genres'] = df_movies['genres'].str.replace('|', ' ').str.strip()
df_movies.head()


# In[6]:


df_movies.info()


# In[7]:


#Remove duplicate Movies if any
df_movies = df_movies.drop_duplicates(subset='movieId', keep='first')


# In[8]:


df_imdb_data.head()


# In[9]:


df_imdb_data['title_cast'] = df_imdb_data['title_cast'].str.replace(' ', '').str.strip()
df_imdb_data['title_cast'] = df_imdb_data['title_cast'].str.replace('|', ' ').str.strip()
df_imdb_data['plot_keywords'] = df_imdb_data['plot_keywords'].str.replace('|', ' ').str.strip()
df_imdb_data['director'] = df_imdb_data['director'].str.replace(' ', '').str.strip()
#df_imdb_data = df_imdb_data.fillna('')


df_imdb_data.head()


# In[10]:


#Link movies with imdb Data
df_movies_imdb = pd.merge(df_movies, df_imdb_data, on='movieId', how='left')


# In[11]:


df_movies_imdb['description_tags'] = (pd.Series(df_movies_imdb[['title','director', 'genres','title_cast','plot_keywords']]
                      .fillna('')
                      .values.tolist()).str.join(' '))

# Convienient indexes to map between book titles and indexes of 
# the books dataframe
movieId = df_movies_imdb['movieId']
indices = pd.Series(df_movies_imdb.index, index=df_movies_imdb['movieId'])

df_movies_imdb['description_tags'] = df_movies_imdb['description_tags'].str.lower()
#df_movies_imdb['description_tags'] = df_movies_imdb['description_tags'].str.replace('[^a-zA-Z\s]', '', regex=True)

df_movies_imdb[['movieId', 'description_tags']]


# In[12]:


#print how many users there are
count_unique_users = df_movies_imdb['movieId'].nunique()
print(count_unique_users)


# In[13]:


# Convert the 'date' column to datetime format
#df_movie_imdb_user['timestamp'] = pd.to_datetime(df_movie_imdb_user['timestamp'], unit='s')

# Find the 5 most recent ratings for each user
#df_movie_imdb_user_top10 = df_movie_imdb_user.groupby('userId').apply(lambda x: x.nlargest(5, 'timestamp')).reset_index(drop=True)


# In[14]:


df_movie_imdb_final = df_movies_imdb[['movieId', 'description_tags']]
df_movie_imdb_final.head()


# In[15]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                     min_df=0, stop_words='english')


# Produce a feature matrix, where each row corresponds to a book,
# with TF-IDF features as columns 
tf_authTags_matrix = tf.fit_transform(df_movie_imdb_final['description_tags'])
tf_authTags_matrix = tf_authTags_matrix.astype(np.float32)


# In[16]:


cosine_sim_authTags = cosine_similarity(tf_authTags_matrix, 
                                        tf_authTags_matrix)
print (cosine_sim_authTags.shape)


# In[17]:


def content_generate_top_N_recommendations(movie_id, N=10):
    # Convert the string book title to a numeric index for our 
    # similarity matrix
    b_idx = indices[movie_id]
    # Extract all similarity values computed with the reference book title
    sim_scores = list(enumerate(cosine_sim_authTags[b_idx]))
    # Sort the values, keeping a copy of the original index of each value
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Select the top-N values for recommendation
    sim_scores = sim_scores[1:N]
    # Collect indexes 
    book_indices = [i[0] for i in sim_scores]
    # Convert the indexes back into titles 
    return movieId.iloc[book_indices]


# In[18]:


content_generate_top_N_recommendations(1, N=10)


# In[19]:


def content_generate_rating_estimate(movie_id, user, rating_data, k=20, threshold=0.0):
    # Convert the book title to a numeric index for our 
    # similarity matrix
    b_idx = indices[movie_id]
    neighbors = [] # <-- Stores our collection of similarity values 
     
    # Gather the similarity ratings between each book the user has rated
    # and the reference book 
    for index, row in rating_data[rating_data['userId']==user].iterrows():
        sim = cosine_sim_authTags[b_idx-1, indices[row['movieId']]-1]
        neighbors.append((sim, row['rating']))
    # Select the top-N values from our collection
    k_neighbors = heapq.nlargest(k, neighbors, key=lambda t: t[0])

    # Compute the weighted average using similarity scores and 
    # user item ratings. 
    simTotal, weightedSum = 0, 0
    for (simScore, rating) in k_neighbors:
        # Ensure that similarity ratings are above a given threshold
        if (simScore > threshold):
            simTotal += simScore
            weightedSum += simScore * rating
    try:
        predictedRating = weightedSum / simTotal
    except ZeroDivisionError:
        # Cold-start problem - No ratings given by user. 
        # We use the average rating for the reference item as a proxy in this case 
        predictedRating = np.mean(rating_data[rating_data['movieId']==movie_id]['rating'])
    return predictedRating


# In[20]:


df_train[df_train['userId'] == 2][3:15]


# In[21]:


df_train.info()


# In[ ]:


results_df = pd.DataFrame(columns=['userId', 'movieId', 'pred_rating'])

# Iterate over the rows of the dataset
for index, row in df_test.iterrows():
    User_ID = row['userId']
    Movie_ID = row['movieId']
    pred_rating = content_generate_rating_estimate(movie_id=Movie_ID, user=User_ID, rating_data=df_train)
    
    # Add the results to the new DataFrame
    results_df = results_df.append({'userId': User_ID, 'movieId': Movie_ID, 'pred_rating': pred_rating}, ignore_index=True)

    
    
    # Iterate over the rows of df_test
for index, row in df_test.iterrows():
    User_ID = row['userId']
    Movie_ID = row['movieId']
    pred_rating = content_generate_rating_estimate(movie_id=Movie_ID, user=User_ID, rating_data=df_train)
    
    # Combine User_ID and Movie_ID into a single column id
    id = f"{User_ID}_{Movie_ID}"
    
    # Add the results to the new DataFrame
    results_df = results_df.append({'id': id, 'pred_rating': pred_rating}, ignore_index=True)

# Print the results DataFrame
print(results_df)


# In[ ]:


# Create an empty list to store the intermediate dataframes
dfs = []
merged_df = pd.DataFrame()

# Set the batch size for creating new dataframes
batch_size = 25000
counter=0
# Iterate over the rows of df_test
for idx, row in enumerate(df_test.itertuples(), start=1):
    User_ID = row.userId
    Movie_ID = row.movieId
    pred_rating = content_generate_rating_estimate(movie_id=Movie_ID, user=User_ID, rating_data=df_train)
    
    # Combine User_ID and Movie_ID into a single column id
    id = f"{User_ID}_{Movie_ID}"
    
    # Create a dictionary to store the row data
    row_data = {'id': id, 'pred_rating': pred_rating}
    
    # Append the row data to the current dataframe
    results_df = pd.DataFrame([row_data])
    
    # Append the current dataframe to the list
    dfs.append(results_df)
    
    # Check if the batch size is reached or if it is the last row
    if idx % batch_size == 0 or idx == len(df_test):
        # Merge the intermediate dataframes into a single dataframe
        merged_df = pd.concat([merged_df] + dfs)
        
        # Do further processing or save the merged dataframe as desired
        
        # Clear the list of dataframes
        dfs = []
        counter = counter + 1
        # Print the current index number
        print(counter)

# Print the merged dataframe
print(merged_df)


# In[ ]:


merged_df.to_csv('submission.csv',index=False)


# In[ ]:


merged_df.tail()


# In[ ]:




