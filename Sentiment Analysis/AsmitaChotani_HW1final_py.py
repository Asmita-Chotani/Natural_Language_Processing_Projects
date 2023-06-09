#!/usr/bin/env python
# coding: utf-8

# # <center>CSCI 544 HOMEWORK 1</center>
# 
# **NAME:** Asmita Chotani
# <br>
# **USC ID:** 3961468036
# <br>

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('punkt')  # for word tokenizing
nltk.download('stopwords') # for determining stop words taht have to be removed
nltk.download('omw-1.4') # for lemmatizing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
import string
import contractions


# In[2]:


get_ipython().system('pip install contractions')


# In[3]:


get_ipython().system(" pip install bs4 # in case you don't have it installed")

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[4]:


df2 = pd.read_csv('./amazon_reviews_us_Beauty_v1_00.tsv', 
                                sep='\t',
                                error_bad_lines=False
                               )
display(df2)
print(df2.columns)


# In[5]:


df3 = pd.read_csv('./amazon_reviews_us_Beauty_v1_00.tsv', 
                                sep='\t',
                                error_bad_lines=False,
                                usecols=["star_rating", "review_body"]
                               )
display(df3)


# In[6]:


# understanding the different rating values that are possible
df3['star_rating'].unique()


# In[7]:


df3['star_rating'].value_counts()


# In[8]:


# creating a copy of the dataframe to work with
df=df3.copy()


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[9]:


# 3 classes are formed for the 5 kinds of ratings possible. 
def categorise(row):  
    if row['star_rating'] ==  1 or row['star_rating']== '1' or row['star_rating'] ==  1.0 :
        return 1
    elif row['star_rating'] ==  2 or row['star_rating']== '2'or row['star_rating'] ==  2.0:
        return 1
    elif row['star_rating'] == 3 or row['star_rating']== '3'or row['star_rating'] ==  3.0:
        return 2
    elif row['star_rating'] ==  4 or row['star_rating']== '4'or row['star_rating'] ==  4.0:
        return 3
    elif row['star_rating'] ==  5 or row['star_rating']== '5'or row['star_rating'] ==  5.0:
        return 3
    else:
        return 0   # the entries with invalid values in the rating column


# In[10]:


df['star_rating'].unique()


# In[11]:


df['class'] = df.apply(lambda row: categorise(row), axis=1)
display(df)


# In[12]:


# understanding the distribution of the classes
df['class'].value_counts()


# In[13]:


df['class'].unique()


# In[14]:


# Creating separate dataframes for separate classes
S1_dfa = df.loc[df['class'] == 1]
S2_dfa = df.loc[df['class'] == 2]
S3_dfa = df.loc[df['class'] == 3]

# COnsidering only 20000 data entries for each class
S1_df=S1_dfa.sample(n=20000)
S2_df=S2_dfa.sample(n=20000)
S3_df=S3_dfa.sample(n=20000)


# In[15]:


# Concatenating 20000 reviews for each class into one dataframe that we will work on
review_df = pd.concat([S1_df, S2_df, S3_df])
display(review_df)


# # Data Cleaning
# 
# 

# ### Reseting Index

# In[16]:


# Since we have randomly chosen 20000 entries from each class, it is necessary to reset the index to avoid 
# repitition of entries.
review_df = review_df.reset_index(drop=True)
display(review_df)


# ### Dealing with Null Values

# In[17]:


# Checking for null values
review_df.isnull().values.any() 


# In[18]:


# Checking number of null values in the two columns
review_df.isnull().sum()


# In[19]:


# Filling the null values with an empty string as only empty value is in the review 
review_df = review_df.fillna('')


# ### Creating New DataFrame to store the length of the reviews after different steps.

# In[20]:


# Creating a separate dataframe to store the length of the reviews after every cleaning, processing step to 
# verify that the task was done successfully
display_df = pd.DataFrame()
display_df['before_cleaning'] = review_df['review_body'].str.len()
display(display_df)


# In[21]:


display(review_df)


# ### Converting into Lower Case

# In[22]:


#Converting the reviews into Lower Case
review_df['review_body'] = review_df['review_body'].str.lower()
display(review_df)


# ### Removing HTML and URLs

# In[23]:


def remove_mention_tag_fn(text):
    text = re.sub(r'@\S*', '', text)
    return re.sub(r'#\S*', '', text)


# In[24]:


# Removing well-formed tags i.e the HTML and URLs
review_df['review_body'] = review_df['review_body'].str.replace(r'<[^<>]*>', '', regex=True) 
review_df['review_body'] = review_df['review_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])


# In[25]:


review_df['review_body'] = review_df['review_body'].apply(remove_mention_tag_fn)
display_df['tag_cleaning'] = review_df['review_body'].str.len()
display(display_df)


# ### Removing punctuations

# In[26]:


def remove_punctuations(text):
    return ''.join(char for char in text if char not in string.punctuation)


# In[27]:


# Remove puctuations
review_df['review_body'] = review_df['review_body'].apply(remove_punctuations)
display_df['punctuation_cleaning'] = review_df['review_body'].str.len()
display(review_df)
display(display_df)


# ### Remove Emojis

# In[28]:


# def remove_emoji_fn(string):
#      emoji_pattern = re.compile('['u'U0001F600-U0001F64F' # emoticons
#      u'U0001F300-U0001F5FF' # symbols & pictographs
#      u'U0001F680-U0001F6FF' # transport & map symbols
#      u'U0001F1E0-U0001F1FF' # flags (iOS)
#      u'U00002702-U000027B0'
#      u'U000024C2-U0001F251'
#      ']+', flags=re.UNICODE)
#     return emoji_pattern.sub(r'', string)


# In[29]:


# review_df['review_body'] = review_df['review_body'].apply(remove_emoji_fn)
# display_df['emoji_cleaning'] = review_df['review_body'].str.len()
# display(review_df)
# display(display_df)


# ### Removing non-alphabets

# In[30]:


def remove_alphanum(text):
    t= " ".join([re.sub('[^A-Za-z]+','', text) for text in nltk.word_tokenize(text)])
    return t


# In[31]:


# Remove non-alpabetics
review_df['review_body']=review_df['review_body'].apply(remove_alphanum)
display_df['alphanum_cleaning'] = review_df['review_body'].str.len()
display(review_df)
display(display_df)


# ### Removing extra spaces

# In[32]:


review_df['review_body'] = review_df['review_body'].apply(lambda x: re.sub(' +', ' ', str(x)))
display_df['remove_spaces'] = review_df['review_body'].str.len()
display(review_df)
display(display_df)


# ### Contracting the words

# In[33]:


def word_contractions(text):
    t=[]
    for i in text.split():
        t.append(contractions.fix(i))
    # Now that the review has been split into a list of words and contracted, the contractions are joined into one sentence
    return ' '.join(t)


# In[34]:


# Contracting the reviews
review_df['review_body']=review_df['review_body'].apply(word_contractions)
# display(review_df)

# # Now that the review has been split into a list of words and contracted, the contractions are joined into one sentence
# review_df['review_body'] = review_df['review_body'].apply(word_contractions)

display_df['post_contractions'] = review_df['review_body'].str.len()

display(review_df)
display(display_df)


# In[35]:


display_df['after_cleaning'] = review_df['review_body'].str.len()
display(display_df)


# In[36]:


print("The length of the reviews initially-  ",display_df['before_cleaning'].mean())
print("The length of the reviews after cleaning-  ", display_df['after_cleaning'].mean())


# # Pre-Processing

# ### Removing the Stop Words

# In[37]:


from nltk.corpus import stopwords


# In[38]:


stop = set(stopwords.words('english'))


# In[39]:


def stop_word_fn(text):
    return ' '.join(i for i in text.split() if i not in (stop))


# In[40]:


review_df['review_body'] = review_df['review_body'].apply(stop_word_fn)
display(review_df)


# In[41]:


display_df['stopword_removal'] = review_df['review_body'].str.len()
display(display_df)


# ### Lemmatization 

# In[42]:


from nltk.stem import WordNetLemmatizer


# In[43]:


wnl = WordNetLemmatizer()
review_df['review_body'] = review_df['review_body'].apply(wnl.lemmatize)
display_df['after_lemmatize'] = review_df['review_body'].str.len()
display(display_df)


# In[44]:


display(review_df)


# In[45]:


review_df['star_rating'].unique()


# In[46]:


review_df['class'].value_counts()


# In[47]:


review_df.isnull().sum()


# In[48]:


print("The length of the reviews post cleaning-  ",display_df['after_cleaning'].mean())
print("The length of the reviews after pre-processing-  ", display_df['after_lemmatize'].mean())


# ## TF-IDF Feature Extraction

# In[49]:


#Splitting the Data into train and test data (split should be of 80%-20%)
Xtrain, Xtest, ytrain, ytest = train_test_split(review_df['review_body'], review_df['class'], train_size=0.80, random_state=4)

print("Training Data Size: ", Xtrain.shape)
print("Testing Data Size: ", Xtest.shape)


# In[50]:


# Verifying the distribution of the classes in the training data
ytrain.value_counts()


# In[51]:


tfID_feat_extract = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=12000
)


# In[52]:


Xtrain_tfid = tfID_feat_extract.fit_transform(Xtrain)

Xtest_tfid = tfID_feat_extract.transform(Xtest)

print("Training document-term matrix : ", Xtrain_tfid)
print("Training feature names for transformation : ", tfID_feat_extract.get_feature_names_out())


# In[53]:


Xtrain_tfid.todense()


# In[54]:


from sklearn.model_selection import GridSearchCV


# # Models

# Grid Search has been used for determining the most efficient hyperparameters for the different models.
# The two types of penalties i.e Ridge and Lasso has been considered for the models and the best is chosen.

# ## Perceptron

# In[55]:


model_perceptron2 = Perceptron(
    alpha=0.00001,
    penalty= 'l2',      #Penalty for wrong prediction
    max_iter=1500,      #Maximum number of iterations 
    shuffle=True,       
    random_state=16, 
    tol=0.001,
)
model_perceptron2=model_perceptron2.fit(Xtrain_tfid , ytrain)
pred_percept2=model_perceptron2.predict(Xtest_tfid)
result2=classification_report(ytest, pred_percept2,output_dict=True)
print(result2)


# In[56]:


i=1
for keys,values in result2.items():
    if i==4:
        i=i+1
        continue
    else:
        print(keys,": ",values['precision'],",",values['recall'],",",values['f1-score'])
        i=i+1


# ## SVM

# In[59]:


svm_model = LinearSVC(
    C=0.35,
    tol=0.001,
    max_iter=1000,                 #Total iterations
    random_state=16,                #Control the random number generation to control the shuffling
    penalty='l1',                  #Norm of Penalty 
    class_weight="balanced",       #Provides the weight to each class
    loss='squared_hinge',          #Specifies the Loss Function
    dual=False,                    #Selects the algorithm to either the dual or primal optimization - l1+square hinge does not work with dual=true
)
svm_model=svm_model.fit(Xtrain_tfid , ytrain)
pred_svm=svm_model.predict(Xtest_tfid)
svm_result=classification_report(ytest, pred_svm,output_dict=True)
print(svm_result)


# In[60]:


i=1
for keys,values in svm_result.items():
    if i==4:
        i=i+1
        continue
    else:
        print(keys,": ",values['precision'],",",values['recall'],",",values['f1-score'])
        i=i+1


# ## Logistic Regression

# In[63]:


lr_model = LogisticRegression(
    C=0.4,
    solver='saga',
    tol=0.01,
    max_iter=2000,               #Max iterations to be considered
    penalty='l2',                #Penalty for wrong prediction 
    multi_class='multinomial',   
    random_state=16,              
)
lr_model = GridSearchCV(lr_model, parameters2)
lr_model=lr_model.fit(Xtrain_tfid , ytrain)
pred_logistic=lr_model.predict(Xtest_tfid)
lr_result=classification_report(ytest, pred_logistic,output_dict=True)

print(lr_result)


# In[64]:


for keys,values in lr_result.items():
    print("Class",keys," ", values)


# In[65]:


i=1
for keys,values in lr_result.items():
    if i==4:
        i=i+1
        continue
    else:
        print(keys,": ",values['precision'],",",values['recall'],",",values['f1-score'])
        i=i+1


# ## Naive Bayes

# In[68]:


nb_model = MultinomialNB(alpha=6)

nb_model=nb_model.fit(Xtrain_tfid , ytrain)
pred_nb=nb_model.predict(Xtest_tfid)
nb_report=classification_report(ytest, pred_nb,output_dict=True)
 
print(nb_report)


# In[69]:


i=1
for keys,values in nb_report.items():
    if i==4:
        i=i+1
        continue
    else:
        print(keys,": ",values['precision'],",",values['recall'],",",values['f1-score'])
        i=i+1


# ## REFERENCES

# In[71]:


# https://www.geeksforgeeks.org/how-to-randomly-select-rows-from-pandas-dataframe/
# https://www.statology.org/pandas-select-rows-based-on-column-values/
# https://stackoverflow.com/questions/45999415/removing-html-tags-in-pandas
# https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
# https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
# https://datatofish.com/lowercase-pandas-dataframe/
# https://stackoverflow.com/questions/39782418/remove-punctuations-in-pandas
# https://www.geeksforgeeks.org/python-map-function/
# https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
# https://aparnamishra144.medium.com/how-to-categorize-a-column-by-applying-a-function-in-pandas-135f47f7ab34
# https://stackoverflow.com/questions/52279834/splitting-training-data-with-equal-number-rows-for-each-classes
# https://www.geeksforgeeks.org/python-remove-unwanted-spaces-from-string/
# https://towardsdatascience.com/primer-to-cleaning-text-data-7e856d6e5791
# https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/
# https://www.programiz.com/python-programming/methods/string/join

