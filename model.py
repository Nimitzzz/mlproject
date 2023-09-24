#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("Housing.csv")


# In[3]:


housing.info()


# In[4]:


housing.head()


# ## Making of the modal

# In[5]:


import numpy as np
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.25,random_state=42)
print(len(train_set))
print(len(test_set))
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

# ## Checking the correlations

# In[6]:


mat=housing.corr()
mat['price'].sort_values(ascending=False)


# In[7]:


a=train_set.drop("price",axis=1)
a_labels=train_set["price"].copy()


# ## Model making cont

# In[8]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(a,a_labels)


# ## TESTING

# In[9]:


tdata=a.iloc[:5]


# In[10]:


tlabels=a_labels.iloc[:5]


# In[11]:


model.predict(tdata)


# In[12]:


list(a_labels)


# ## saving the model

# In[13]:


import pickle
pickle.dump(model, open('model.pkl','wb'))


# ## usage example

# In[14]:


from joblib import dump,load
import numpy as np
model = pickle.load(open('model.pkl','rb'))
check=np.array([[8000,4,4,5,3]])
model.predict(check)

