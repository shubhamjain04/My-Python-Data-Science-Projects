# BANK REVIEWS AND COMPLAINTS ANALYSIS 

# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

path = 'C:/Users/shubh/Desktop/ALABS PYTHON CASE STUDIES/FINAL/5. Bank Reviews-Complaints Analysis/'

# IMPORTING DATASETS 
raw_data = pd.read_excel(path + 'BankReviews.xlsx')

raw_data.head()

### EXPLORATORY DATA ANALYSIS

# CHECKING DATA TYPE OF VARIABLES
raw_data.info()

# COUNT OF DIFFERENT STAR RATINGS
raw_data.Stars.value_counts().plot(kind = 'bar', figsize = (15, 5))
plt.show()

raw_data.Stars.value_counts()

# NUMBER OF REVIEWS FOR EACH BANK
raw_data.BankName.value_counts()

# AVERAGE STARS FOR EACH BANK
raw_data[['BankName', 'Stars']].groupby('BankName').agg('mean')

# COUNT OF STARS FOR EACH BANK
pd.crosstab(index = raw_data['BankName'], columns = raw_data['Stars'], values = raw_data['Stars'], aggfunc = 'count')

# TIME PERIOD OF DATA
raw_data.Date.min(), raw_data.Date.max()

# CHECKING FOR MISSING VALUES
raw_data.isnull().sum()

# DATA PROCESSING STEPS

# CONVERTING THE DATA TO LOWER CASE
raw_data.loc[:, 'Reviews'] = raw_data.loc[:, 'Reviews'].str.lower()
raw_data.loc[:, 'Reviews'].head()

# REMOVING SPECIAL CHARACTERS ['\n', '\t', '\r', '&']
spec_char = ['\n', '\t', '\r', '&']

for char in spec_char :
    raw_data.loc[:, 'Reviews'] = raw_data.loc[:, 'Reviews'].str.replace(char, '')
    
raw_data.loc[:, 'Reviews'].head()

# REMOVING PUNCTUATION MARKS
punc_char = ["'", '.', ',', '-', '_', '?', '/', '!', '@', '(', ')', ';', '"', ':']

for char in punc_char :
    raw_data.loc[:, 'Reviews'] = raw_data.loc[:, 'Reviews'].str.replace(char, '')
    
raw_data.loc[:, 'Reviews'].head()

# REMOVING NUMBERS
num_char = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

for char in num_char :
    raw_data.loc[:, 'Reviews'] = raw_data.loc[:, 'Reviews'].str.replace(char, '')
    
raw_data.loc[:, 'Reviews'].head()

raw_data.head()

# GETTING A POLARITY SCORE FOR EACH REVIEW/FEEDBACK USING TEXTBLOB
from textblob import TextBlob as tb

Polarity_Score = pd.Series(dtype = 'float')

for i in range(0, len(raw_data.loc[:, 'Reviews'])) :
    Polarity_Score = Polarity_Score.append(pd.Series(tb(raw_data.loc[i, 'Reviews']).polarity))

Polarity_Score.reset_index(drop = True, inplace = True)

# CREATING A NEW FEATURE POLARITY SCORE FOR EACH REVIEW
raw_data.loc[:, 'Polarity_Score'] = Polarity_Score
raw_data.head()

# REVIEWS WHOSE STARS ARE 5 BUT POLARITY SCORE IS NEGATIVE
raw_data.loc[(raw_data.loc[:, 'Stars'] == 5) & (raw_data.loc[:, 'Polarity_Score'] <= 0)].shape

# REVIEWS WHOSE STARS ARE 1 BUT POLARITY SCORE IS +VE
raw_data.loc[(raw_data.loc[:, 'Stars'] == 1) & (raw_data.loc[:, 'Polarity_Score'] > 0)].shape

# GETTING  A LIST OF STOPWORDS TO BE REMOVED FROM OUR DATA
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['wyndham', 'capital', 'mortgage', 'american', 'first', 'reliance', 'inc', 'rate', 'guaranteed', 'triumph', 
                   'lending', 'pacific', 'north', 'us', 'give'])
print(stop_words)

# TOKENIZING
raw_data.loc[:, 'Reviews_Tokenized'] = raw_data.loc[:, 'Reviews'].str.split(' ')
raw_data.head()

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

# REMOVING STOPWORDS AND LEMMATIZING WORDS
cleaned_text = []

for text in raw_data.loc[:, 'Reviews_Tokenized'] :
    cleaned_tokens = []
    for word in text :
        word = wnl.lemmatize(word, pos = 'v')
        if word not in stop_words and word != '' :
            cleaned_tokens.append(word)
            
    cleaned_text.append(cleaned_tokens)
    
print(cleaned_text)      

# REPLACING TOKENIZED REVIEWS WITH CLEANED TEXT
raw_data.loc[:, 'Reviews_Tokenized'] = cleaned_text
raw_data.head()

# REPLACING REVIEWS WITH CLEANED AND LEMMATIZED REVIEW(Review_Tokenized as string) 
for i in range(0, len(raw_data.loc[:, 'Reviews_Tokenized'])) :
    raw_data.loc[i, 'Reviews'] = ' '.join(raw_data.loc[i, 'Reviews_Tokenized'])

raw_data.head()

# CREATING A FEATURE POLARITY_SCORE2 TO GET POLARITY OF CLEANED REVIEW

Polarity_Score2 = pd.Series(dtype = 'float')

for i in range(0, len(raw_data.loc[:, 'Reviews'])) :
    Polarity_Score2 = Polarity_Score2.append(pd.Series(tb(raw_data.loc[i, 'Reviews']).polarity))

Polarity_Score2.reset_index(drop = True, inplace = True)

raw_data.loc[:, 'Polarity_Score2'] = Polarity_Score2
raw_data.head()

raw_data.loc[(raw_data.loc[:, 'Stars'] == 5) & (raw_data.loc[:, 'Polarity_Score2'] <= 0)].shape

raw_data.loc[(raw_data.loc[:, 'Stars'] == 1) & (raw_data.loc[:, 'Polarity_Score2'] > 0)].shape

# SINCE THE NUMBER OF MISCLASSIFIED REVIEWS i.e. POSITIVE REVIEWS GETTING A NEGATIVE SENTIMENT SCORE AND VICE-VERSA ARE LOWER IN POLARITY_SCORE1, WE WILL BE USING IT TO CREATE OUR SENTIMENT FEATURE.

# SENTIMENT CLASSIFICATION OF REVIEWS

# USING POLARITY SCORE TO CREATE SENTIMENT COLUMN, 
# IF POLARITY > 0 : SENTIMENT --> +VE(1)
# IF POLARITY < 0 : SENTIMENT --> -VE(-1)
# IF POLARITY = 0 : SENTIMENT --> NEUTRAL(0)
raw_data.loc[:, 'Sentiment'] = np.where(raw_data.loc[:, 'Polarity_Score'] > 0, 'Positive', 
                                        np.where(raw_data.loc[:, 'Polarity_Score'] < 0, 'Negative', 'Neutral'))

raw_data.head()

raw_data.loc[:, 'Sentiment'].value_counts().plot(kind = 'bar', figsize = (15, 5))
plt.show()

# IDENTIFYING TOP/FREQUENT +VE AND -VE WORDS
positive_reviews = raw_data.loc[raw_data.loc[:, 'Stars'] == 5, 'Reviews_Tokenized']
positive_reviews.reset_index(drop = True, inplace = True)

positive_reviews

negative_reviews = raw_data.loc[raw_data.loc[:, 'Stars'] == 1, 'Reviews_Tokenized']
negative_reviews.reset_index(drop = True, inplace = True)

negative_reviews

# GETTING INDIVIDUAL WORDS OUT OF POSITIVE REVIEWS TO BUILD A WORD CLOUD
word_series = pd.Series(dtype = 'object')

for i in range(0, len(positive_reviews)) :
    for j in range(0, len(positive_reviews[i])) :
        word_series = word_series.append(pd.Series(positive_reviews[i][j]))

word_series.nunique()

(word_series.value_counts() >= 40).sum()

freq_pos_words = word_series.value_counts()[0:50]
freq_pos_words

# CONVERTING FREQ_POS_WORDS IN A STRING
freq_pos_words = '\n'.join(freq_pos_words.index)
freq_pos_words

# GETTING INDIVIDUAL WORDS OUT OF NEGATIVE REVIEWS TO BUILD A WORD CLOUD
word_series = pd.Series(dtype = 'object')

for i in range(0, len(negative_reviews)) :
    for j in range(0, len(negative_reviews[i])) :
        word_series = word_series.append(pd.Series(negative_reviews[i][j]))

word_series.nunique()

word_series.value_counts()

(word_series.value_counts() > 20).sum()


freq_neg_words = word_series.value_counts()[0:50]
freq_neg_words

# CONVERTING FREQ_NEG_WORDS IN A STRING
freq_neg_words = '\n'.join(freq_neg_words.index)
freq_neg_words

from wordcloud import WordCloud

# BUILDING A WORDCLOUD OF FREQUENT WORDS FROM BOTH POSITIVE AND NEGATIVE REVIEWS

# WORD CLOUD FOR POSITIVE REVIEWS
wordcloud = WordCloud(background_color = 'white', min_font_size = 5).generate(freq_pos_words) 

fig = plt.figure(figsize=(15, 8))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

# WORDCLOUD FOR NEGATIVE REVIEWS
wordcloud = WordCloud(background_color = 'white', min_font_size = 5).generate(freq_neg_words) 

fig = plt.figure(figsize=(15, 8))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

# FROM THE WORD CLOUD OF POSITIVE REVIEWS WE CAN CONCLUDE THAT CUSTOMERS ARE HAPPY WITH SERVICES BECAUSE OF THE APPEARENCE OF WORDS LIKE ***** GREAT, RESPONSIVE, BEST, EASY, THANK, KNOWLEDGEABLE, HELPFUL *****.

# FROM THE WORD CLOUD OF NEGATIVE REVIEWS WE CAN CONCLUDE THAT CUSTOMERS ARE UNHAPPY WITH SERVICES BECAUSE OF THE APPEARENCE OF WORDS LIKE ***** POOR, LATER, NEVER, CLOSE *****
# FURTHER WE CAN CONCLUDE THAT MOST CUSTOMERS ARE UNHAPPY BECAUSE OF UNTIMELY DELIVERY OF SERVICES,(SINCE TIME IS COMING AS FREQUENT WORD ALONG WITH LATER AND NEVER, HENCE WE ARE ABLE TO ARRIVE AT THIS CONCLUSION).

# PREDICTING STAR RATING FOR REVIEWS
raw_data.head()

# CREATING TF-IDF MATRIX
from sklearn.feature_extraction.text import TfidfVectorizer

# SELECTING ONLY TOP 250 FEATURES
tf_idf = TfidfVectorizer(max_features = 250)

tf_idf_df = tf_idf.fit_transform(raw_data['Reviews'])
tf_idf_df

tf_idf_df = pd.DataFrame(tf_idf_df.toarray(), columns = tf_idf.get_feature_names())
tf_idf_df.head()

tf_idf_df.shape

tf_idf_df['Stars'] = raw_data['Stars']

tf_idf_df.head()

# GETTING A LIST OF ALL THE INDEPENDENT FEATURES
independent_variables = tf_idf_df.columns.difference(['Stars'])
independent_variables

# SPLITTING THE DATA INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tf_idf_df[independent_variables], tf_idf_df['Stars'], test_size = 0.3, random_state = 123, shuffle = True)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

y_train.value_counts()

y_test.value_counts()

# BUILDING A SUPPORT VECTOR CLASSIFIER TO MODEL OUR DATA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# INITIALIZING AN INSTANCE OF A SVC CLASS
svc_clf = SVC( class_weight = 'balanced', gamma = 'auto',  kernel = 'rbf')

# DEFINING THE PARAMETER GRID 
param_grid = {'C': [20, 21, 22, 23, 24, 25, 26]}

# GRID SEARCH TO IDENTIFY BEST PARAMETERS
svc_clf_model = GridSearchCV( estimator = svc_clf, 
                              param_grid = param_grid, 
                              cv = 5, 
                              scoring = 'accuracy',
                              n_jobs = -1, 
                              verbose = 10 ).fit(  x_train, y_train )

svc_clf_model.best_estimator_

svc_clf_model.best_score_

# BUILDING SUPPORT VECTOR CLASSIFIER WITH BEST PARAMETERS
svc_clf = SVC( C = 23, class_weight = 'balanced', gamma = 'auto',  kernel = 'rbf').fit(x_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TRAINING ACCURACY
print(accuracy_score(svc_clf.predict(x_train), y_train))

# TESTING ACCURACY
print(accuracy_score(svc_clf.predict(x_test), y_test))

# CONFUSION MATRIX FOR TRAINING DATASET
sns.heatmap(confusion_matrix(svc_clf.predict(x_train), y_train), annot = True)

# CLASSIFICATION REPORT FOR TRAINING DATASET
print(classification_report(svc_clf.predict(x_train), y_train))

# CONFUSION MATRIX FOR TESTING DATASET
sns.heatmap(confusion_matrix(svc_clf.predict(x_test), y_test), annot = True)

# CLASSIFICATION REPORT FOR TESTING DATASET
print(classification_report(svc_clf.predict(x_test), y_test))

# TOPIC MODELING, CLUSTERING TO IDENTIFY KEY THEMES OF PROBLEMS
raw_data.head()

negative_reviews = raw_data.loc[raw_data['Stars'] == 1, 'Reviews']
negative_reviews

# CREATING A TF-IDF MATRIX
tf_idf2 = TfidfVectorizer()

tf_idf_df2 = tf_idf2.fit_transform(negative_reviews)
tf_idf_df2

tf_idf_df2 = pd.DataFrame(tf_idf_df2.toarray(), columns = tf_idf2.get_feature_names())
tf_idf_df2.head()

# USING LDA FOR TOPIC MODELLING
from sklearn.decomposition import LatentDirichletAllocation

lda_tm = LatentDirichletAllocation(n_components = 5, n_jobs = -1)

lda_model = lda_tm.fit(tf_idf_df2)
lda_model

terms = tf_idf_df2.columns

# FUNCTION FOR PRINTING TERMS FOR EACH TOPIC
def selected_topics(model, vectorizer, top_n = 20):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(terms[i], str(round(topic[i],2))+"%") for i in topic.argsort()[ : -top_n - 1 : -1]]) 

selected_topics(lda_model, tf_idf_df2)

# THOUGH THE TOPICS ARE NOT VERY EXCLUSIVE OF EACH BUT STILL WE CAN MAKE FOLLOWING INFERENCES :

# 1. CUSTOMERS ARE COMPLAINING ABOUT SLOW AND VAGUE DOCUMENTATION FOR LOAN APPROVAL.
# 2. CUSTOMERS CMPLAINING ABOUT POOR COMMUNICATION PROCESS.
# 3. CUSTOMERS ARE COPLAINING ABOUT PROCESSING TIME AND PROCESSING FEES.
