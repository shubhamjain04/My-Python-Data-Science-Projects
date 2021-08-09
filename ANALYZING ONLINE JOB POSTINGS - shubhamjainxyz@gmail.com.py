# ANALYZING ONLINE JOB POSTINGS CASE STUDY - shubhamjainxyz@gmail.com

# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SETTING DISPLAY OPTIONS
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
path = 'C:/Users/shubh/Desktop/ALABS PYTHON CASE STUDIES/FINAL/6. Analyzing online Job Postings/'

# IMPORTING DATASET
raw_data = pd.read_csv(path + 'data job posts.csv')

raw_data.info()

# SELECTING ONLY USEFUL FEATURES
raw_data2 = raw_data[['JobDescription', 'JobRequirment', 'AboutC', 'Year', 'Month', 'IT']]

raw_data2.head()

raw_data2.shape

raw_data2.columns

# REMOVING ALL OBSERVATIONS WHERE JOB DESCRIPTION IS NOT AVAILABLE
x = raw_data2.loc[raw_data['JobDescription'].isnull() | raw_data['JobRequirment'].isnull() | raw_data['AboutC'].isnull()].index

raw_data2.drop(index = x, inplace = True)

raw_data2.shape

raw_data2.index = range(0, raw_data2.shape[0])

raw_data2.head()

# OBJECTIVE 1 : IDENTIFYING THE NATURE OF JOBS THAT ARE IN DEMAND IN ARMENIA

# APPLYING DATA CLEANING STEPS

# CONVERTING ALL TEXT TO LOWER CASE
raw_data2.loc[:, 'JobDescription'], raw_data2.loc[:, 'JobRequirment'], raw_data2.loc[:, 'AboutC'] = raw_data2.loc[:, 'JobDescription'].str.lower(), raw_data2.loc[:, 'JobRequirment'].str.lower(), raw_data2.loc[:, 'AboutC'].str.lower()

raw_data2.head()

# REMOVING SPECIAL CHARACTERS ['\n', '\t', '\r']
spec_char = ['\n', '\t', '\r', '&']

for char in spec_char :
    raw_data2.loc[:, 'JobDescription'] = raw_data2.loc[:, 'JobDescription'].str.replace(char, '')
    raw_data2.loc[:, 'AboutC'] = raw_data2.loc[:, 'AboutC'].str.replace(char, '')
    raw_data2.loc[:, 'JobRequirment'] = raw_data2.loc[:, 'JobRequirment'].str.replace(char, '')

# REMOVING PUNCTUATION MARKS
punc_char = ["'", '.', ',', '-', '_', '?', '/', '!', '@', '(', ')', ';']

for char in punc_char :
    raw_data2.loc[:, 'JobDescription'] = raw_data2.loc[:, 'JobDescription'].str.replace(char, '')
    raw_data2.loc[:, 'AboutC'] = raw_data2.loc[:, 'AboutC'].str.replace(char, '')
    raw_data2.loc[:, 'JobRequirment'] = raw_data2.loc[:, 'JobRequirment'].str.replace(char, '')

# REMOVING NUMBERS
num_char = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

for char in num_char :
    raw_data2.loc[:, 'JobDescription'] = raw_data2.loc[:, 'JobDescription'].str.replace(char, '')
    raw_data2.loc[:, 'AboutC'] = raw_data2.loc[:, 'AboutC'].str.replace(char, '')
    raw_data2.loc[:, 'JobRequirment'] = raw_data2.loc[:, 'JobRequirment'].str.replace(char, '')

# TOKENIZING
raw_data2.loc[:, 'JobDescription'] = raw_data2.loc[:, 'JobDescription'].str.split(' ')
raw_data2.loc[:, 'AboutC'] = raw_data2.loc[:, 'AboutC'].str.split(' ')
raw_data2.loc[:, 'JobRequirment'] = raw_data2.loc[:, 'JobRequirment'].str.split(' ')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# REMOVING STOPWORDS
stop_words = stopwords.words('english')
stop_words.extend(['armenia', 'usa', 'look', 'candidate', 'visit', 'company', 'llc', 'ínternational', 'world', 'armenian', 'cjsc', 'provide', 'seek', 'please', 'yerevan', 'responsible', 'incumbent'])
print(stop_words)

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# LEMMATIZATION TO RULE OUT DIFFERENT FORMS OF SAME WORDS
wnl = WordNetLemmatizer()

cleaned_text = []

for text in raw_data2.loc[:, 'JobDescription'] :
    cleaned_tokens = []
    for word in text :
        word = wnl.lemmatize(word, pos = 'v')
        if word not in stop_words and word != '' :
            cleaned_tokens.append(word)
            
    cleaned_text.append(cleaned_tokens)
    
print(cleaned_text)    

# REPLACING THE VALUES OF JOBDESCRIPTION WITH CLEANED TEXT
raw_data2.loc[:, 'JobDescription'] = pd.Series(cleaned_text)

# REPLACING NAN VALUES WITH 'is', SINCE THEY ARE CAUSING PROBLEMS IN ITERATION
raw_data2.loc[:, 'AboutC'].fillna('is', inplace = True)

cleaned_text = []

for text in raw_data2.loc[:, 'AboutC'] :
    cleaned_tokens = []
    for word in text :
        word = wnl.lemmatize(word, pos = 'v')
        if word not in stop_words and word != '' :
            cleaned_tokens.append(word)
            
    cleaned_text.append(cleaned_tokens)
    
print(cleaned_text)

# REPLACING THE VALUES OF AboutCompany(ABOUTC) WITH CLEANED TEXT
raw_data2.loc[:, 'AboutC'] = pd.Series(cleaned_text)

# REPLACING NAN VALUES WITH 'is', SINCE THEY ARE CAUSING PROBLEMS IN ITERATION
raw_data2.loc[:, 'JobRequirment'].fillna('is', inplace = True)

cleaned_text = []

for text in raw_data2.loc[:, 'JobRequirment'] :
    cleaned_tokens = []
    for word in text :
        word = wnl.lemmatize(word, pos = 'v')
        if word not in stop_words and word != '' :
            cleaned_tokens.append(word)
            
    cleaned_text.append(cleaned_tokens)
    
print(cleaned_text)

# REPLACING THE VALUES OF JOB REQUIREMENT WITH CLEANED TEXT
raw_data2.loc[:, 'JobRequirment'] = pd.Series(cleaned_text)

# GETTING A LIST OF INDIVIDUAL ELEMENTS FROM JOB DESCRIPTION COLUMN
word_series = pd.Series(dtype = 'object')

for i in range(0, len(raw_data2.loc[:, 'JobDescription'])) :
    for j in range(0, len(raw_data2.loc[i, 'JobDescription'])) :
        word_series = word_series.append(pd.Series(raw_data2.loc[i, 'JobDescription'][j]))

word_series

word_series.nunique()

(word_series.value_counts() >= 500).sum()

freq_words_jd = word_series.value_counts()[0:94]

# FREQUENT WORDS IN JOB DESCRIPTION TO USED FOR MAKING WORD CLOUD
freq_words_jd

freq_words_jd_list = '\n'.join(freq_words_jd.index)
freq_words_jd_list

# GETTING A LIST OF INDIVIDUAL ELEMENTS FROM COMPANY DESCRIPTION COLUMN
word_series = pd.Series(dtype = 'object')

for i in range(0, len(raw_data2.loc[:, 'AboutC'])) :
    for j in range(0, len(raw_data2.loc[i, 'AboutC'])) :
        word_series = word_series.append(pd.Series(raw_data2.loc[i, 'AboutC'][j]))

word_series

word_series.nunique()

word_series.value_counts()

(word_series.value_counts() >= 200).sum()

freq_words_cd = word_series.value_counts()[0:174]

# FREQUENT WORDS IN COMPANY DESCRIPTION TO BE USED FOR BUILDING A WORD CLOUD
freq_words_cd

freq_words_cd_list = '\n'.join(freq_words_cd.index)
freq_words_cd_list

pip install wordcloud
from wordcloud import WordCloud

# BUILDING A WORD CLOUD FOR JOB DESCRIPTION
wordcloud_jd = WordCloud(background_color ='white',
                         min_font_size = 5).generate(freq_words_jd_list)
plt.figure(figsize = (25, 10))
plt.imshow(wordcloud_jd)
plt.show()

# BUILDING A WORD CLOUD FOR COMPANY DESCRIPTION
wordcloud_cd = WordCloud(background_color ='white',
                         min_font_size = 5).generate(freq_words_cd_list)
plt.figure(figsize = (25, 10))
plt.imshow(wordcloud_cd)
plt.show()

# CONCLUSION FOR OBJECTIVE 1.
IT IS EVIDENT FROM BOTH THE WORD CLOUDS THAT MOST OF THE JOBS IN ARMENIA ARE IN THE FOLLOWING SECTORS IN THE ORDER PRESENTED :

1. PRODUCT BASED COMPANIES
2. SOFTWARE COMPANIES
3. FINANCE COMPANIES
4. TECHNOLOGY COMPANIES
5. BANKING 
6. MANAGEMENT

JOB ROLES THAT ARE IN DEMAND ARE :

1. ENGINEERS
2. DEVELOPERS
3. MANAGERS
4. ASSISTANTS
5. DIRECTORS

## OBJECTIVE 2 : IDENTIFYING THE SKILL SETS IN DEMAND

# GETTING A LIST OF INDIVIDUAL ELEMENTS FROM JOB REQUIREMENT COLUMN
word_series = pd.Series(dtype = 'object')

for i in range(0, len(raw_data2.loc[:, 'JobRequirment'])) :
    for j in range(0, len(raw_data2.loc[i, 'JobRequirment'])) :
        word_series = word_series.append(pd.Series(raw_data2.loc[i, 'JobRequirment'][j]))

word_series

word_series.nunique()

word_series.value_counts()

(word_series.value_counts() > 500).sum()

freq_words_jr = word_series.value_counts()[0:309]

freq_words_jr_list = '\n'.join(freq_words_jr.index)

# BUILDING A WORD CLOUD FOR JOB REQUIREMENT
wordcloud_jr = WordCloud(background_color ='white',
                         min_font_size = 5).generate(freq_words_jr_list)
plt.figure(figsize = (25, 10))
plt.imshow(wordcloud_jr)
plt.show()

# CONCLUSION FOR OBJECTIVE 2.
# FROM THE WORD CLOUD MADE FOR JOB REQUIREMENT FOLLOWING CONCLUSIONS CAN BE MADE REGARDING SKILLS IN DEMAND :

# 1. Customer centric skills are the ones which are in most demand like customer support, customer relationship management.
# 2. Management based skills are the second most in demand skills like product management, project management, programming.
# 3. Software based skills are the third most in demand skills like application development/maintenance, system development

# Apart from these top most skills, skills like planning and designning, operations managemnet, technical skills, data analysis, financial skills are also in demand.

## OBJECTIVE 3 : CLASSIFYING A GIVEN JOBPOST AS IT OR NON-IT BASED ON JOB/COMPANY DESCRIPTION

raw_data2.head()

raw_data2.IT.value_counts()

raw_data2.loc[:, 'IT_encoded'] = raw_data2.loc[:, 'IT'].map({ False : 0, True : 1 })

raw_data2.head()

raw_data3 = raw_data2.loc[:, ['JobDescription', 'AboutC', 'IT_encoded']]

raw_data3.head()

# TRANSFORMING JOBDESCRIPTION AND ABOUTC FEATURES INTO STRINGS SO THAT TF-IDF CAN BE CREATED

# CREATING A NEW FEATURE  JobDescription_str CONTAINING STRING FORM OF TOKENIZED JobDescription
for i in range(0, len(raw_data3.loc[:, 'JobDescription'])) :
    raw_data3.loc[i, 'JobDescription_str'] = ' '.join(raw_data3.loc[i, 'JobDescription'])

# CREATING A NEW FEATURE  AboutC_str CONTAINING STRING FORM OF TOKENIZED AboutC
for i in range(0, len(raw_data3.loc[:, 'AboutC'])) :
    raw_data3.loc[i, 'AboutC_str'] = ' '.join(raw_data3.loc[i, 'AboutC'])

raw_data3.head()

raw_data3 = raw_data3.loc[:, ['JobDescription_str', 'AboutC_str', 'IT_encoded']]
raw_data3.head()

# CREATING A TF-IDF MATRIX
from sklearn.feature_extraction.text import TfidfVectorizer

# CREATING TF-IDF FOR JobDescription_str FEATURE
tf_idf_jd  = TfidfVectorizer(max_features = 100)

tf_idf_df_jd = tf_idf_jd.fit_transform(raw_data3['JobDescription_str'])
tf_idf_df_jd

tf_idf_df_jd = pd.DataFrame(tf_idf_df_jd.toarray(), columns = tf_idf_jd.get_feature_names())
tf_idf_df_jd.head()

# CREATING TF-IDF FOR JobDescription_str FEATURE
tf_idf_ac  = TfidfVectorizer(max_features = 50)

tf_idf_df_ac = tf_idf_ac.fit_transform(raw_data3['AboutC_str'])
tf_idf_df_ac

tf_idf_df_ac = pd.DataFrame(tf_idf_df_ac.toarray(), columns = tf_idf_ac.get_feature_names())
tf_idf_df_ac.head()

# CONCATENATING tf_idf_df_jd AND tf_idf_df_ac 
final_model_data = pd.concat([tf_idf_df_jd, tf_idf_df_ac], axis = 1)
final_model_data

final_model_data.loc[:, 'IT'] = raw_data3.loc[:, 'IT_encoded']

final_model_data.head()

# GETTING FEATURES NAMES
features = final_model_data.columns.difference(['IT'])

# SPLITTING THE DATA INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(final_model_data[features], final_model_data['IT'], test_size = 0.3, random_state = 123)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

y_train.value_counts()

y_test.value_counts()

# BUILDING A NAIVE BAYES MODEL FOR OUR DATA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

naive_bayes_model = GaussianNB()
naive_bayes_model

naive_bayes_fit = naive_bayes_model.fit(x_train, y_train)
naive_bayes_fit

# ACCURACY SCORE FOR TRAINING AND TESTING
print('Training Accuracy : ', accuracy_score(y_train, naive_bayes_fit.predict(x_train)))
print('Testing Accuracy : ', accuracy_score(y_test, naive_bayes_fit.predict(x_test)))

# CONFUSION MATRIX FOR TRAINING DATASET
sns.heatmap(confusion_matrix(y_train, naive_bayes_fit.predict(x_train)), annot = True)

# CONFUSION MATRIX FOR TESTING DATASET
sns.heatmap(confusion_matrix(y_test, naive_bayes_fit.predict(x_test)), annot = True)

# BUILDING A SVM CLASSIFIER
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc_model = SVC(random_state = 123)

param_grid = { 'C' : [0.1, 1, 1.5, 2, 2.5]}

svc_fit = GridSearchCV(estimator = svc_model, 
                       param_grid = param_grid,
                       cv = 5,
                       scoring = 'accuracy',
                       n_jobs = -1,
                       verbose = 10).fit(x_train, y_train)

svc_fit.best_score_

svc_fit.best_estimator_

# BUILDING SVC WITH BEST PARAMETERS
svc_fit = SVC(C = 0.5, random_state = 123).fit(x_train, y_train)

# TRAINING AND TESTING ACCURACY
print('Training Accuracy : ', accuracy_score(svc_fit.predict(x_train), y_train))
print('Testing Accuracy : ', accuracy_score(svc_fit.predict(x_test), y_test))

# BUILDING A RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state = 123)

param_grid = {'n_estimators' : [100, 110, 120, 130, 140]}

rf_fit = GridSearchCV(estimator = rf_model,
                      param_grid = param_grid,
                      cv = 5,
                      scoring = 'accuracy',
                      n_jobs = -1,
                      verbose = 10).fit(x_train, y_train)

rf_fit.best_score_

rf_fit.best_params_

# BUILDING A RF MODEL WITH BEST ESTIMATORS
rf_fit = RandomForestClassifier(n_estimators = 60, random_state=123).fit(x_train, y_train)

# TRAINING AND TESTING ACCURACY
print('Training Accuracy : ', accuracy_score(rf_fit.predict(x_train), y_train))
print('Testing Accuracy : ', accuracy_score(rf_fit.predict(x_test), y_test))

# BUILDING A KNN CLASSIFIER FOR OUR DATA
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()

param_grid = {'n_neighbors' : range(2, 10)}

knn_fit = GridSearchCV(estimator = knn_model,
                       param_grid = param_grid,
                       cv = 5,
                       scoring = 'accuracy',
                       n_jobs = -1,
                       verbose = 5).fit(x_train, y_train)

knn_fit.best_score_

knn_fit.best_params_

# BUILDING  KNN MODEL WITH BEST PARAMETERS
knn_fit = KNeighborsClassifier(n_neighbors = 4).fit(x_train, y_train)

# TRAINING AND TESTING ACCURACY
print('Training Accuracy : ', accuracy_score(knn_fit.predict(x_train), y_train))
print('Testing Accuracy : ', accuracy_score(knn_fit.predict(x_test), y_test))


# CONCLUSION FOR OBJECTIVE 3.
# TO CLASSIFY A GIVEN JOB AS IT OR NON-IT BASED ON JOB AND COMPANY DESCRIPTION I BUILT FOUR DIFFERENT MODELS i.e. 

# 1. NAIVE BAYES - IT IS GIVING VERY LOW ACCURACY COMPARED TO OTHER MODELS.

# 2. SUPPORT VECTOR CLASSIFIER - GIVES AN ACCURACY OF 93% FOR TRAINING DATASET AND 91.3% FOR TESTING DATASET.

# 3. RANDOM FOREST CLASSIFIER - THIS MODEL IS GETTING OVERFITTED.

# 4. KNN - GIVES AN ACCURACY OF 92% FOR TRAINING DATASET AND 90% FOR TESTING DATASET.

# BASED ON THE OBSERVATIONS OF ALL THE MODELS WE CHOSE SUPPORT VECTOR CLASSIFIER AS OUR FINAL MODEL.

## OBJECTIVE 4 : GIVEN A JOB TITLE IDENTIFY TOP 5 SIMILAR JOBS BASED ON JOB POST 

raw_data4 = raw_data['jobpost']
raw_data4

# APPLYING DATA CLEANING STEPS TO RAW_DATA3

# CONVERTING ALL TEXT TO LOWER CASE
raw_data4 = raw_data4.str.lower()
raw_data4

# REMOVING SPECIAL CHARACTERS ['\n', '\t', '\r']
spec_char = ['\n', '\t', '\r', '&']

for char in spec_char :
    raw_data4 = raw_data4.str.replace(char, '')

# REMOVING PUNCTUATION MARKS
punc_char = ["'", '.', ',', '-', '_', '?', '/', '!', '@', '(', ')', ';', ':']

for char in punc_char :
    raw_data4 = raw_data4.str.replace(char, '')

# REMOVING NUMBERS
num_char = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

for char in num_char :
    raw_data4 = raw_data4.str.replace(char, '')

# TOKENIZING
raw_data4 = raw_data4.str.split(' ')

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# REMOVING STOPWORDS
stop_words = stopwords.words('english')
stop_words.extend(['armenia', 'usa', 'look', 'candidate', 'visit', 'company', 'llc', 'ínternational', 'world', 'armenian', 'cjsc', 'provide', 'seek', 'please', 'yerevan', 'responsible', 'incumbent'])
print(stop_words)


# LEMMATIZATION TO RULE OUT DIFFERENT FORMS OF SAME WORDS
wnl = WordNetLemmatizer()

cleaned_text = []

for text in raw_data4 :
    cleaned_tokens = []
    for word in text :
        word = wnl.lemmatize(word, pos = 'v')
        if word not in stop_words and word != '' :
            cleaned_tokens.append(word)
            
    cleaned_text.append(cleaned_tokens)
    
# print(cleaned_text)    

# REPLACING THE VALUES OF JOB POST WITH CLEANED TEXT
raw_data4 = pd.Series(cleaned_text)
raw_data4

# JOINING TOKENIZED DATASET TO GET OUT FINAL DATASET
for i in range(0, len(raw_data4)) :
    raw_data4[i] = ' '.join(raw_data4[i])

raw_data4

# IMPORTING TF-IDF VECTORIZER TO CREATE A TF-IDF MATRIX FOR JOB POSTS
from sklearn.feature_extraction.text import TfidfVectorizer

# SELECTING ONLY TOP 1000 FEATURES FOR BETTER TIME EFFIENCY
tf_idf_jp = TfidfVectorizer(max_features = 1000)

# CREATING TF-IDF
tf_idf_df_jp = tf_idf_jp.fit_transform(raw_data4)
tf_idf_df_jp

# COVERTING TF-IDF MATRIX TO A DATAFRAME AND ASSIGNING PROPER FEATURE NAMES
tf_idf_df_jp = pd.DataFrame(tf_idf_df_jp.toarray(), columns = tf_idf_jp.get_feature_names())
tf_idf_df_jp.head()

tf_idf_df_jp.shape

# SETTING A JOB TITLE FOR CALCULATING SIMILARITY
job_title = 'Software Engineer'

tf_idf_jt = TfidfVectorizer()

# CREATING A TF-IDF MATRIX OF JOB TITLE
tf_idf_df_jt = tf_idf_jt.fit_transform(pd.Series(job_title))
tf_idf_df_jt

# CREATING A DATAFRAME OF TF-IDF MATRIX
tf_idf_df_jt = pd.DataFrame(tf_idf_df_jt.toarray(), columns = tf_idf_jt.get_feature_names())
tf_idf_df_jt.head()

# GETTING FEATURE NAMES OF JOB TITLE SINCE ONLY THESE FEATURES WILL USED FROM TF-IDF OF JOB POST FOR CALCULATING DISTANCE/SIMILARITY
job_features = tf_idf_df_jt.columns


# IMPORTING COSINE DISTANCE METRIC
from sklearn.metrics.pairwise import cosine_distances

# GETTING THE INDEXES OF TOP 5 JOBS AS CALCULATED BY COSINE DISTANCE METRIC
top_5_jobs = pd.DataFrame(cosine_distances(tf_idf_df_jt, tf_idf_df_jp[job_features])).T.sort_values(by = 0).head().index

raw_data.loc[top_5_jobs, 'jobpost']
