#!/usr/bin/env python
# coding: utf-8

# # <font style="color:#008fff;">Machine Learning Modeling</font>
# <hr>

# In[1]:


import pandas as pd
import numpy as np
import time
import os
import sklearn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import sys
import random
import pickle

#Disabling Warnings
warnings.filterwarnings('ignore')

# to make this notebook's output stable across runs
random.seed(42)


# ## <font style="color:#008fff;">Reading in preprocessed dataset</font>

# In[2]:


def loadDataset(file_name):
    df = pd.read_csv(file_name)
    return df

start_time= time.time()
df_train_preprocessed = loadDataset("Dataset/preprocessed_data.csv")
print("***Elapsed time to read csv files --- %s seconds ---***" % (time.time() - start_time))


# In[3]:


df_train_preprocessed.head(10)


# ### Split our dataset into X_train and y_train

# In[4]:


X_train = df_train_preprocessed.drop('label', axis=1)
y_train = df_train_preprocessed['label']


# In[5]:


X_train.shape, y_train.shape


# ## <font style="color:#008fff;">Feature Selection</font>
# 
# There are a handful of feature selection methods in Scikit-Learn with classification. According to Sklearn's documentation, common feature selection algorithms include `chi2`, `f_classif`, and `mutual_info_classif` (https://scikit-learn.org/stable/modules/feature_selection.html)
#  - We will try out the 3 we see in this documentation to see which features are most commonly chosen
# 
# **Other Potential Feature Selection methods to experiment in the future: Mean Absolute Difference, Fisher Score, different method from scratch**

# In[23]:


# Importing feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


# ### Feature selection using Sklearn's chi-squared:

# In[20]:


# Taking the top 5 most best descriptive features using chi-squared testing
chi2_selector = SelectKBest(chi2, k=5)
X_kbest_chi2 = chi2_selector.fit_transform(X_train, y_train)

print(f'Out of {X_train.shape[1]} features in our original dataset, we get the top {X_kbest_chi2.shape[1]} chosen by chi2')

# Showing which columns chi2 has chosen
selected_features_chi2 = chi2_selector.get_support(indices=True)
print(f'Top 5 features selected using chi-squared: {list(X_train.iloc[:, selected_features_chi2].columns)}')


# ### Feature selection using Sklearn's f_classif

# In[22]:


# Taking the top 5 best descriptive feature using f classification
f_classif_selector = SelectKBest(f_classif, k=5)
X_kbest_f_classif = f_classif_selector.fit_transform(X_train, y_train)

print(f'Out of {X_train.shape[1]} features in our original dataset, we get the top {X_kbest_f_classif.shape[1]} chosen by f_classif')

# Showing which columns f_classif has chosen
selected_features_f_classif = f_classif_selector.get_support(indices=True)
print(f'Top 5 features selected using f_classif: {list(X_train.iloc[:, selected_features_f_classif].columns)}')


# ### Feature selection using Sklearn's mutual_info_classif

# In[24]:


# Taking the top 5 best descriptive feature using mutual info classif
MIC_selector = SelectKBest(mutual_info_classif, k=5)
X_kbest_MIC = MIC_selector.fit_transform(X_train, y_train)

print(f'Out of {X_train.shape[1]} features in our original dataset, we get the top {X_kbest_MIC.shape[1]} chosen by mutual_info_classif')

# Showing which columns f_classif has chosen
selected_features_MIC = MIC_selector.get_support(indices=True)
print(f'Top 5 features selected using f_classif: {list(X_train.iloc[:, selected_features_MIC].columns)}')


# ### It seems that all 3 methods from scikit-learn all chose 'who_is', 'https', 'profanity_score_prob', 'js_len_scaled',  and 'js_obf_len_scaled' as their top 5 features. We will be using this for now.

# In[27]:


X_kbest_features = X_kbest_chi2
X_kbest_features


# ## <font style="color:#008fff;">ML Modeling: K-Nearest Neighbors</font>

# ### Building KNN Model for the FULL feature set (X_train):

# In[28]:


from sklearn.neighbors import KNeighborsClassifier

knn_full = KNeighborsClassifier(n_neighbors=3)
knn_full.fit(X_train.values, y_train.values)


# In[29]:


# Save the model to disk
knn_full_filename = 'Models/knn_full_features.sav'
pickle.dump(knn_full, open(knn_full_filename, 'wb'))


# In[30]:


# load the model from disk
knn_full = pickle.load(open(knn_full_filename, 'rb'))


# ### Building KNN Model for the feature set after FEATURE SELECTION (X_kbest_features):

# In[31]:


# KNN with reduced features
knn_reduced = KNeighborsClassifier(n_neighbors=3)
knn_reduced.fit(X_kbest_features, y_train.values)


# In[32]:


# Save the model to disk
knn_reduced_filename = 'Models/knn_reduced_features.sav'
pickle.dump(knn_reduced, open(knn_reduced_filename, 'wb'))


# In[33]:


# load the model from disk
knn_reduced = pickle.load(open(knn_full_filename, 'rb'))


# ## <font style="color:#008fff;">ML Modeling: (Gaussian) Naive Bayes</font>

# ### Building Naive Bayes Model for the FULL feature set:

# In[34]:


from sklearn.naive_bayes import GaussianNB

gnb_full = GaussianNB()
gnb_full.fit(X_train.values, y_train.values)


# In[35]:


# Save the model to disk
gnb_full_filename = 'Models/gnb_full_features.sav'
pickle.dump(gnb_full, open(gnb_full_filename, 'wb'))


# In[36]:


# load the model from disk
gnb_full = pickle.load(open(gnb_full_filename, 'rb'))


# ### Building Naive Bayes Model for the feature set after FEATURE SELECTION:

# In[37]:


gnb_reduced = GaussianNB()
gnb_reduced.fit(X_train.values, y_train.values)


# In[38]:


# Save the model to disk
gnb_reduced_filename = 'Models/gnb_reduced_features.sav'
pickle.dump(gnb_reduced, open(gnb_reduced_filename, 'wb'))


# In[39]:


# load the model from disk
gnb_reduced = pickle.load(open(gnb_reduced_filename, 'rb'))


# ## <font style="color:#008fff;">ML Modeling: Decision Tree</font> 

# ### Building Decision Tree Model for the FULL feature set:

# In[40]:


from sklearn.tree import DecisionTreeClassifier
dc_full = DecisionTreeClassifier(max_depth=3)
dc_full.fit(X_train.values, y_train.values)


# In[41]:


#Save the model to disk
dc_full_filename = 'Models/dc_full_features.sav'
pickle.dump(dc_full, open(dc_full_filename, 'wb'))


# In[42]:


# load the model from disk
dc_full = pickle.load(open(dc_full_filename, 'rb'))


# ### Building Random Forest Model for feature set after FEATURE SELECTION

# In[43]:


dc_reduced = DecisionTreeClassifier(max_depth=3)
dc_reduced.fit(X_train.values, y_train.values)


# In[44]:


#Save the model to disk
dc_reduced_filename = 'Models/dc_reduced_features.sav'
pickle.dump(dc_reduced, open(dc_reduced_filename, 'wb'))


# In[45]:


# load the model from disk
dc_reduced = pickle.load(open(dc_reduced_filename, 'rb'))


# ## <font style="color:#008fff;">ML Modeling: Random Forest</font> 

# ### Building Random Forest Model for the FULL feature set:

# In[46]:


from sklearn.ensemble import RandomForestClassifier

rfc_full = RandomForestClassifier(n_estimators=100, random_state=100) #Random_state is a seeded value and n_estimators are the n amount of trees
rfc_full.fit(X_train.values, y_train.values)


# In[47]:


#Save the model to disk
rfc_full_filename = 'Models/rfc_full_features.sav'
pickle.dump(rfc_full, open(rfc_full_filename, 'wb'))


# In[48]:


# load the model from disk
rfc_full = pickle.load(open(rfc_full_filename, 'rb'))


# ### Building Random Forest Model for feature set after FEATURE SELECTION

# In[49]:


rfc_reduced = RandomForestClassifier(n_estimators=100, random_state=100) #Random_state is a seeded value and n_estimators are the n amount of trees
rfc_reduced.fit(X_train.values, y_train.values)


# In[50]:


#Save the model to disk
rfc_reduced_filename = 'Models/rfc_reduced_features.sav'
pickle.dump(rfc_reduced, open(rfc_reduced_filename, 'wb'))


# In[51]:


# load the model from disk
rfc_reduced = pickle.load(open(rfc_reduced_filename, 'rb'))


# ## <font style="color:#008fff;">Testing/Performance Evaluation</font> 

# In[ ]:


# Testing and performance evaluation goes here

