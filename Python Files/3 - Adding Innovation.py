#!/usr/bin/env python
# coding: utf-8

# # <font style="color:#008fff;">Adding Innovation</font>
# <hr>

# ### This part demonstrates the innovative twist we will be implementing. For every sample URL that is classified as malicious, we want to compare its domain name to domain names of another dataset purely of legitimate URL's found here: https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset?resource=download
# 
# ### By doing this, we can predict whether the malicious URL is disguising as another legitimate entity, which is a common practice done by cybercriminals to trick victims to clicking into a malicious link

# In[189]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import spacy

import time
import os
import warnings
import sys
import random
import pickle
import collections


import warnings
from tld import get_tld
import time

#Disabling Warnings
warnings.filterwarnings('ignore')


# ## <font style="color:#008fff;">Reading in dataframe of legitimate URL's and preprocessing it:</font>

# In[16]:


legit_companies = pd.read_csv('Dataset/Company Names/companies_sorted.csv')


# In[17]:


legit_companies.head()


# In[26]:


legit_domain_names = legit_companies[['name', 'domain']]


# In[27]:


legit_domain_names


# Dropping NA values:

# In[28]:


legit_domain_names.isnull().sum()


# In[29]:


print(f'Before {len(legit_domain_names)}')
legit_domain_names = legit_domain_names.dropna()
print(f'After {len(legit_domain_names)}')


# In[31]:


legit_domain_names


# In[41]:


def extract_domain_names(url):
    fragment = url.split('.')
    return fragment[0]


# In[43]:


domain_names = legit_domain_names['domain'].map(extract_domain_names)


# In[51]:


legit_domain_names['domain no tld'] = list(domain_names)


# In[55]:


legit_domain_names.drop('domain', axis=1, inplace=True)


# ## <font style="color:#008fff;">Dataset of Malicious and Benign Webpages</font>

# ### Reading in a sample of datapoints from main Dataset of Malicious and Benign Webpages (Our original dataset we've used to preprocess, train, test on the previous 2 notebooks), which we will perform classification on with models trained and tested on notebook 2

# In[76]:


# PREPROCESSING HELPER FUNCTIONS
from profanity_check import predict_prob, predict
from urllib.parse import urlparse
from tld import get_tld

# Getting rid of outliers using clamp transformation
def find_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    
    for index, val in df.iteritems():
        if val < (q1 - 1.5 * IQR): # Small outliers below lower quartile
            df[index] = (q1 - 1.5 * IQR)
        elif val > (q3 + 1.5 * IQR): # Large outliers above upper quartile
            df[index] = (q3 + 1.5 * IQR)

    return df

# If tld == gov, then is_gov_tld = 1, else gov_tld = 0
def make_gov_column(df):
    gov_col = []
    for index, val in df.iteritems():
        if val == 'gov':
            gov_col.append(1)
        else:
            gov_col.append(0)
    return np.array(gov_col)


def clean_url(url):
    url_text=""
    try:
        domain = get_tld(url, as_object=True)
        domain = get_tld(url, as_object=True)
        url_parsed = urlparse(url)
        url_text= url_parsed.netloc.replace(domain.tld," ").replace('www',' ') +" "+ url_parsed.path+" "+url_parsed.params+" "+url_parsed.query+" "+url_parsed.fragment
        url_text = url_text.translate(str.maketrans({'?':' ','\\':' ','.':' ',';':' ','/':' ','\'':' '}))
        url_text.strip(' ')
        url_text.lower()
    except:
        url_text = url_text.translate(str.maketrans({'?':' ','\\':' ','.':' ',';':' ','/':' ','\'':' '}))
        url_text.strip(' ')
    return url_text

def predict_profanity(url_cleaned):
    arr=predict_prob(url_cleaned.astype(str).to_numpy())
    arr= arr.round(decimals=3)
    #df['url_vect'] = pd.DataFrame(data=arr,columns=['url_vect'])
    return arr

def preprocess(df_current):
    df = df_current.copy()
    
    start_time= time.time()

    # ------------ Address outliers via clamp transformation --------------
    url_len_clamped = df['url_len'].copy()
    url_len_clamped = find_outliers_IQR(url_len_clamped)
    js_len_clamped = df['js_len'].copy()
    js_len_clamped = find_outliers_IQR(js_len_clamped)
    js_obf_len_clamped = df['js_obf_len'].copy()
    js_obf_len_clamped = find_outliers_IQR(js_obf_len_clamped)
    
    df['url_len'] = url_len_clamped
    df['js_len'] = js_len_clamped
    df['js_obf_len'] = js_obf_len_clamped
    
    # --------------- Scaling numerical features ---------------
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    url_len_scaled = scaler.fit_transform(df[['url_len']])
    df['url_len_scaled'] = url_len_scaled

    js_len_scaled = scaler.fit_transform(df[['js_len']])
    df['js_len_scaled'] = js_len_scaled

    js_obf_len_scaled = scaler.fit_transform(df[['js_obf_len']])
    df['js_obf_len_scaled'] = js_obf_len_scaled
    
    
    # ---------------- Binary Encoding for Categorical Attributes ------------------
    identifyWho_Is = {'incomplete': 0, 'complete': 1}
    df['who_is'] = [identifyWho_Is[item] for item in df.who_is]
    
    identifyHTTPS = {'no': 0, 'yes': 1}
    df.https = [identifyHTTPS[item] for item in df.https]
    
    # --------------- Handling TLD Column -------------------------
    gov_binary_val = make_gov_column(df['tld'])
    df.insert(2, column = "is_gov_tld", value=gov_binary_val)
    
    
    # ---------------- Probabilty based profanity score on text columnsk ------------------
    from profanity_check import predict_prob, predict
    profanity_score_prob = predict_prob(np.array(df['content']))
    df.insert(5, column='profanity_score_prob', value=profanity_score_prob)
    
    
    # ------------------ Cleaning URL's --------------------
    url_cleaned = df['url'].map(clean_url)
    df.insert(1, column='url_cleaned', value=url_cleaned)
    url_vect = predict_profanity(df['url_cleaned'])
    df.insert(2, column='url_vect', value=url_vect)
    
    # ---------------------- Preprocess labels into binary values ----------------------
    identifyLabels = {'bad': 1, 'good': 0}
    df['label'] = [identifyLabels[item] for item  in df.label]
    
    # ------------ Drop Unecessary Columns, or Original Columns after preprocessing that still remain -------------
    df.drop(['geo_loc', 'ip_add', 'url_len', 'js_len', 'js_obf_len', 'tld', 'content', 'url', 'url_cleaned'], axis=1, inplace=True)
    
    # ---------------------- Rearrange Columns ----------------------
    titles = ['url_vect', 'is_gov_tld', 'who_is', 'https', 'profanity_score_prob', 
              'url_len_scaled', 'js_len_scaled','js_obf_len_scaled',
              'label'] # Same order as our training data

    df = df[titles] 
    
    print("***Elapsed time preprocess --- %s seconds ---***" % (time.time() - start_time))
    return df


# In[77]:


def loadDataset(file_name, idx_col=False):
    start_time= time.time()
    if idx_col:
        df = pd.read_csv(file_name, index_col=[0])
    else:
        df = pd.read_csv(file_name)
    print("***Elapsed time to read csv files --- %s seconds ---***" % (time.time() - start_time))
    return df

df_test = loadDataset("Dataset/Webpages_Classification_test_data.csv", idx_col=True)


# In[78]:


df_test = df_test.sample(13627)
df_test_preprocessed = preprocess(df_test)


# ### NOTE: We want to keep original test data because for any datapoint in our testing set that is considered malicious by our machine learning models, we extract its domain name, then compare it to domain names from our dataframe of legitimate URLs

# In[86]:


# Original test data
X_test = df_test.drop('label', axis=1)
y_test = df_test['label']

# Preprocessed version of test data
X_test_preprocessed = df_test_preprocessed.drop('label', axis=1)
y_test_preprocessed = df_test_preprocessed['label']


# ## <font style="color:#008fff;">Making Our Predictions</font>

# ### Reading in our models (In this notebook, we will just be testing on our optimized models trained on reduced features since they have better accuracy)

# In[85]:


knn_reduced_filename = 'Models/Optimized/knn_reduced_features_opt.sav' # Reduced feature set
knn_reduced = pickle.load(open(knn_reduced_filename, 'rb'))

gnb_reduced_filename = 'Models/Optimized/gnb_reduced_features_opt.sav'
gnb_reduced = pickle.load(open(gnb_reduced_filename, 'rb'))

dc_reduced_filename = 'Models/Optimized/dc_reduced_features_opt.sav'
dc_reduced = pickle.load(open(dc_reduced_filename, 'rb'))

rfc_reduced_filename = 'Models/Optimized/rfc_reduced_features_opt.sav'
rfc_reduced = pickle.load(open(rfc_reduced_filename, 'rb'))


# In[183]:


# Function to have all 4 models make a majority vote
def vote_predictions(row):
    reduced_features_1 = ['who_is', 'https', 'profanity_score_prob', 'js_len_scaled', 'js_obf_len_scaled']
    knn_input = [row.loc[reduced_features_1]]
    
    reduced_features_2 = ['url_vect', 'is_gov_tld', 'js_obf_len_scaled']
    gnb_dc_rfc_input = [row.loc[reduced_features_2]]
    
    # Each models predicts
    preds = []
    preds.append(knn_reduced.predict(knn_input)[0])
    preds.append(gnb_reduced.predict(gnb_dc_rfc_input)[0])
    preds.append(dc_reduced.predict(gnb_dc_rfc_input)[0])
    preds.append(rfc_reduced.predict(gnb_dc_rfc_input)[0])
    
    vote_counts = collections.Counter(preds)
    
    if vote_counts[0] > vote_counts[1]:
        return 0
    elif vote_counts[0] < vote_counts[1]:
        return 1
    else: # If tie, randomly choose either one
        return random.choice([0, 1])


start_time = time.time()

majority_preds = []
potentially_risky_urls = []
for index, row in X_test_preprocessed.iterrows():
    # Take majority vote
    vote = vote_predictions(row)
    majority_preds.append(vote)
    
    if vote == 1: # If majority vote == 1, means URL with majority vote from all 1 models is potentially risky
        potentially_risky_urls.append(X_test.loc[index]['url']) # Getting raw URL of risky URL
    
print("***Elapsed time to make predictions --- %s seconds ---***" % (time.time() - start_time))


# In[184]:


print(f'Accuracy: {accuracy_score(majority_preds,y_test_preprocessed)}')
print(f'Precision: {precision_score(majority_preds,y_test_preprocessed)}')
print(f'Recall: {recall_score(majority_preds,y_test_preprocessed)}')
print(f'F1: {f1_score(majority_preds,y_test_preprocessed)}')
print(f'AUC: {roc_auc_score(majority_preds,y_test_preprocessed)}')


# In[188]:


print(f'Numebr of URLs classified as malicious according to our 4 models\'s votes: {len(potentially_risky_urls)}')


# ## <font style="color:#008fff;">Measuring Word Similarity Using Edit Distance</font>

# Edit distance (AKA Levenshtein Distance), is a measure of the minimum number of operations (Insert, delete, and replace) required to transform one string to another. For example, consider strings "kitten" and "sitting". To transform "kitten" into "sitting":
#  - Substitute 'k' with 's'
#  - Substitute 'e' with 'i'
#  - Insert 'i' before 't'
#  - Substitute 'n' with 'g'

# In[196]:


# Function taking in 2 strings to return the edit distance. (DYNAMIC PROGRAMMING)
def calculate_edit_distance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    # Create a 2D matrix to store the edit distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Compute the edit distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Return the edit distance between the two strings
    return dp[len1][len2]


# In[197]:


calculate_edit_distance('amazon', 'amzon')


# In[210]:


# Extract domain name from our list of potentially risky urls
risky_urls_domains = []
for url in potentially_risky_urls:
    fragments = clean_url(url).split()
    risky_urls_domains.append(fragments[0])


# **NOTE: We will only be sampling to 22,000 legitimate URLs to make our comparison. Originally, we had ~5 million, which took too much time to calculate edit distance as seen in the next couple cells**

# In[264]:


safe_urls_domains = list(legit_domain_names['domain no tld'])[:30000]


# ## Calculate Edit Distance

# In[265]:


results = {'Risky Domain': [],
           'Shortest Edit Distance': [],
           'Potentially Disguising As': []
          }

start_time = time.time()

for i in risky_urls_domains:
    edit_dist_recorded = {}
    for j in safe_urls_domains:
        ed = calculate_edit_distance(i, j)
        edit_dist_recorded[j] = ed # Getting all edit distances 
    
    # With the corresponding i, find which j it has the shortest edit distance with
    j_with_lowest_dist = min(edit_dist_recorded, key=lambda k: edit_dist_recorded[k])
    lowest_dist = edit_dist_recorded[j_with_lowest_dist]
    
    # Appending to results
    results['Risky Domain'].append(i)
    results['Shortest Edit Distance'].append(lowest_dist)
    results['Potentially Disguising As'].append(j_with_lowest_dist)
    
print("***Elapsed time to make predictions --- %s seconds ---***" % (time.time() - start_time))


# In[266]:


results = pd.DataFrame(results)


# ### We filtered all malicious domain names where the edit distance is less than 3, and here are our results. Some of these malicious domain names may not actually be disguising as a legitimate domain, but this gives insights of what cybercriminals can do to trick users by pretending to be a legitimate entity domain.

# In[267]:


results[results['Shortest Edit Distance'] < 3]

