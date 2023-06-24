#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(2)


# In[6]:


credits.head()


# In[7]:


movies['title'].head()


# In[8]:


credits['cast'].head()


#   ### Merging both the databases on the basis of title

# In[9]:


movies.shape


# In[10]:


credits.shape


# In[11]:


movies = movies.merge(credits, on = 'title')


# `total 24 coloums should be there but title will come once it wont repeat

# In[12]:


movies.head(1)


# ### Data Preprocessing

# We will remove those coloums or attributes which do not contribute to the
# recommendation of a movie, eg : budget, homepage

# #### Important coloumns
# 
# genres ,
# id ,
# keywords ,
# title ,
# overview ,
# cast, crew
#  

# In[13]:


movies.info()


# In[14]:


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[15]:


movies.head()


# In[16]:


movies.isnull().sum()


# In[17]:


movies.dropna(inplace = True)


# In[18]:


movies.isnull().sum()


# In[19]:


movies.shape


# In[20]:


movies.duplicated().sum()


# In[21]:


movies.iloc[0].genres


# In[ ]:





# but the thing is that indices are strings so we must convert them to integer

# In[22]:


import ast
ast.literal_eval


# In[23]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[24]:


movies['genres'] = movies['genres'].apply(convert)


# In[25]:


movies


# In[26]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[27]:


movies['keywords']


# In[28]:


movies['cast']


# In[29]:


# def convert3(obj):
#     L = []
#     counter = 0
#     for i in ast.literal_eval(obj):
#         if(counter != 3):
#             L.append(i['name'])
#             counter += 1
#         else:
#             break
#     return L

# # for movie cast we are only concerned with the top 3 actors


# In[30]:


# movies['cast'].apply(convert3)


# In[31]:


movies.head()


# In[32]:


movies['crew'][0]


# In[33]:


def fetch_director(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# for movie cast we are only concerned with the top 3 actors


# In[34]:


movies['crew'].apply(fetch_director)


# In[35]:


movies['crew']= movies['crew'].apply(fetch_director)


# In[36]:


movies.head()


# In[37]:


movies['overview'][0]


# In[38]:


movies['overview'] = movies['overview'].apply(lambda x : x.split())


# In[39]:


movies['overview'][0]


# In[40]:


movies['genres'].apply(lambda x : [i.replace(" ", "")for i in x])
#REMOVING SPACES IN WORDS FOR GENRE COLOUMN


# In[41]:


movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ", "")for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ", "")for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ", "")for i in x])


# In[42]:


movies['cast']


# In[43]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[44]:


movies.head()


# In[45]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[46]:


new_df


# In[47]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[48]:


new_df['tags'][0]


# In[49]:


new_df['tags'].apply(lambda x : x.lower())


# In[50]:


new_df.head()


# ### Text Vectorization

# Similarity would be assumed and calculated on the basis of Cosine Similarity of 2 vectors

# In[51]:


from sklearn.feature_extraction.text import CountVectorizer


# In[52]:


cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[53]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[54]:


vectors


# In[55]:


cv.get_feature_names()


# In[56]:


#stemming and lemmatization


# In[57]:


get_ipython().system('pip install nltk')
import nltk


# In[58]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[59]:


ps.stem('adorable')


# In[60]:


ps.stem('adore')


# In[61]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[62]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy Science Fiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron'
)


# In[63]:


new_df['tags'].apply(stem)


# In[64]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[65]:


cv.get_feature_names()


# In[66]:


from sklearn.metrics.pairwise import cosine_similarity


# In[67]:


similarity = cosine_similarity(vectors)


# In[68]:


similarity[0]


# In[69]:


similarity[1]


# In[70]:


similarity[2]


# In[71]:


similarity = cosine_similarity(vectors)


# In[72]:


sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x : x[1])[1:6]    


# In[73]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x : x[1])[1:6]    
    
    for i in movies_list:
#         print(i[0])
        print(new_df.iloc[i[0]].title)


# In[74]:


recommend('Avatar')


# In[75]:


new_df.iloc[1216].title


# In[76]:


recommend('Batman Begins')


# In[77]:


import pickle


# In[80]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))


# In[81]:


pickle.dump(similarity, open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




