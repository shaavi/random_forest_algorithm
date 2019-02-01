#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the library with the iris dataset 
from sklearn.datasets import load_iris 
# Loading scikit's random forest classifier library 
from sklearn.ensemble import RandomForestClassifier 
# Loading pandas 
import pandas as pd 
# Loading numpy 
import numpy as np 
# Setting random seed 
np.random.seed(0)


# In[4]:


# Creating an object called iris with the iris data 
iris = load_iris() 
# print iris
# Creating a dataframe with the four feature variables 
df = pd.DataFrame(iris.data, columns=iris.feature_names) 
# Viewing the top 5 rows 
df.head()


# In[8]:



# Adding a new column for the species name 
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) 
# Viewing the top 5 rows 
df.head()


# In[9]:


# Creating Test and Train Data 
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75 
# View the top 5 rows 
df.head()


# In[10]:


# Creating dataframes with test rows and training rows 
train, test = df[df['is_train']==True], df[df['is_train']==False] 
# Show the number of observations for the test and training dataframes 
print('Number of observations in the training data:', len(train)) 
print('Number of observations in the test data:',len(test))


# In[13]:


# Create a list of the feature column's names 
features = df.columns[:4] 
# View features 
features


# In[14]:


# Converting each species name into digits 
y = pd.factorize(train['speertcies'])[0] 
# Viewing target 
y


# In[15]:


# Creating a random forest Classifier. 
clf = RandomForestClassifier(n_jobs=2, random_state=0) 
# Training the classifier 
clf.fit(train[features], y)


# In[16]:


# Applying the trained Classifier to the test 
clf.predict(test[features])


# In[21]:


# Viewing the predicted probabilities of the first 10 observations 
clf.predict_proba(test[features])[10:20]


# In[24]:


# mapping names for the plants for each predicted plant class 
preds = iris.target_names[clf.predict(test[features])] 
# View the PREDICTED species for the first five observations 
preds[0:5]


# In[23]:


# Viewing the ACTUAL species for the first five observations 
test['species'].head()


# In[25]:


# Creating confusion matrix 
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species']) 


# In[28]:


preds = iris.target_names[clf.predict( [[5.0, 3.6, 1.4, 2.0], [5.0, 3.6, 1.4, 2.0]] )]
preds


# In[ ]:




