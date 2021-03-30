# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:42:49 2021

@author: Agnij
"""

import pandas as pd
import string
translator = str.maketrans('', '', string.punctuation)
import re
from tqdm import trange

df_sub = pd.read_csv('sample_submission.csv')
df_train = pd.read_csv('train_file.csv')
df_test = pd.read_csv('test_file.csv')

df_train.dropna(inplace = True)
df_train.reset_index(inplace = True)
## TARGET VARIABLE IS SENTIMENT SCORE

'''

EDA ->
1.Feature Engineering -> Popularity as a single feature among all platforms i.e. highest value among the three


TOPIC -> ECONOMY,  highest popularity score on facebook has zero popularity on G+ and LinkedIn
fb = df_train_main.groupby(['Facebook'])['Topic'].unique()

df_train_main.iloc[(df_train_main.index[df_train_main['Facebook'] == 49211].tolist())[0]]

df_train_main['Title'][(df_train_main.index[df_train_main['Facebook'] == 49211].tolist())[0]]
df_train_main['Headline'][(df_train_main.index[df_train_main['Facebook'] == 49211].tolist())[0]]


TOPIC -> ECONOMY,  highest popularity score on G+, has relevant popularity on other platforms

gplus = df_train_main.groupby(['GooglePlus'])['Topic'].unique()
df_train_main.iloc[(df_train_main.index[df_train_main['GooglePlus'] == 1267].tolist())[0]]

TOPIC -> ECONOMY,  highest popularity score on LinkedInhas zero popularity for Facebook, and a meager 18 for G+

ln = df_train_main.groupby(['LinkedIn'])['Topic'].unique()
df_train_main.iloc[(df_train_main.index[df_train_main['LinkedIn'] == 3716].tolist())[0]]



#df_train.drop(['SentimentHeadlineAbs','SentimentTitleAbs'], axis=1, inplace =True)

df_train['SentimentHeadlineAbs'] = df_train['SentimentHeadline'].apply(lambda x: -1 if x<0 else x)
df_train['SentimentHeadlineAbs'] = df_train['SentimentHeadlineAbs'].apply(lambda x: 1 if x>0 else x)
df_train['SentimentHeadlineAbs'] = df_train['SentimentHeadlineAbs'].apply(lambda x: 0 if x==0 else x)

df_train['SentimentTitleAbs'] = df_train['SentimentTitle'].apply(lambda x: -1 if x<0 else x)
df_train['SentimentTitleAbs'] = df_train['SentimentTitleAbs'].apply(lambda x: 1 if x>0 else x)
df_train['SentimentTitleAbs'] = df_train['SentimentTitleAbs'].apply(lambda x: 0 if x==0 else x)

## Reduction of News consumption on Weekends ##
df_train.groupby(['publishMonth'])['SentimentTitleAbs'].value_counts().plot(kind='barh')

## Either no data or News consumption reduced to nil bw April to October, The former being liklier ##
df_train.groupby(['publishMonth'])['SentimentHeadlineAbs'].value_counts().plot(kind='barh')

## VADER SENTIMENT ANALYZER LEXICON UPDATE WORD ANALYSIS
## BASED ON TOPIC AND ABSOLUTE VALUE OF SENTIMENT
## SELECTED WORDS HAVE RANGE -4 TO 4

from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

obama_title_pos  = ''
obama_title_neg  = ''

obama_headline_pos = ''
obama_headline_neg = ''

economy_title_pos  = ''
economy_title_neg  = ''

economy_headline_pos = ''
economy_headline_neg = ''

microsoft_title_pos  = ''
microsoft_title_neg  = ''

microsoft_headline_pos = ''
microsoft_headline_neg = ''

palestine_title_pos  = ''
palestine_title_neg  = ''

palestine_headline_pos = ''
palestine_headline_neg = ''

#review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

for i in trange(0,len(new)):
    ## OBAMA
    if new['Topic'][i] == 2 and new['SentimentTitleAbs'][i] == 1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['first','ap','said','friday','thursday','monday','republican','state','washington','said','barack','obama','obamas','president','white','house','us','gop','gun','guns','says','plan','calls','state','climate']]
        obama_title_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 2 and new['SentimentTitleAbs'][i] == -1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['first','ap','said','friday','thursday','monday','republican','state','washington','said','barack','obama','obamas','president','white','house','us','gop','gun','guns','says','plan','calls','state','climate']]
        obama_title_neg += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 2 and new['SentimentHeadlineAbs'][i] == 1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['first','ap','said','friday','thursday','monday','republican','state','washington','said','barack','obama','obamas','president','white','house','us','gop','gun','guns','says','plan','calls','state','climate']]
        obama_headline_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 2 and new['SentimentHeadlineAbs'][i] == -1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['first','ap','said','friday','thursday','monday','republican','state','washington','said','barack','obama','obamas','president','white','house','us','gop','gun','guns','says','plan','calls','state','climate']]
        obama_headline_neg += ' '+ ' '.join(s)
        
        
    ## ECONOMY
    if new['Topic'][i] == 0 and new['SentimentTitleAbs'][i] == 1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['economy','economic','us','year','said','new','percent','government','world','chinese','china','chinas','could','says','global']]
        economy_title_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 0 and new['SentimentTitleAbs'][i] == -1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['economy','economic','us','year','said','new','percent','government','world','chinese','china','chinas','could','says','global']]
        economy_title_neg += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 0 and new['SentimentHeadlineAbs'][i] == 1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['economy','economic','us','year','said','new','percent','government','world','chinese','china','chinas','could','says','global']]
        economy_headline_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 0 and new['SentimentHeadlineAbs'][i] == -1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['economy','economic','us','year','said','new','percent','government','world','chinese','china','chinas','could','says','global']]
        economy_headline_neg += ' '+ ' '.join(s)
        
        
    ## MICROSOFT
    if new['Topic'][i] == 1 and new['SentimentTitleAbs'][i] == 1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['microsoft','microsofts','windows','said','says','new','one','company','lumia']]
        microsoft_title_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 1 and new['SentimentTitleAbs'][i] == -1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['microsoft','microsofts','windows','said','says','new','one','company','lumia']]
        microsoft_title_neg += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 1 and new['SentimentHeadlineAbs'][i] == 1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['microsoft','microsofts','windows','said','says','new','one','company','lumia']]
        microsoft_headline_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 1 and new['SentimentHeadlineAbs'][i] == -1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['microsoft','microsofts','windows','said','says','new','one','company','lumia']]
        microsoft_headline_neg += ' '+ ' '.join(s)
        
    ## PALESTINE
    if new['Topic'][i] == 3 and new['SentimentTitleAbs'][i] == 1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['palestine','palestinian','israel','israeli','said','state','palestinians','ramallah','east','monday','west','un']]
        palestine_title_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 3 and new['SentimentTitleAbs'][i] == -1:
        s = [w for w in new['Title'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['palestine','palestinian','israel','israeli','said','state','palestinians','ramallah','east','monday','west','un']]
        palestine_title_neg += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 3 and new['SentimentHeadlineAbs'][i] == 1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['palestine','palestinian','israel','israeli','said','state','palestinians','ramallah','east','monday','west','un']]
        palestine_headline_pos += ' '+ ' '.join(s)
        
    if new['Topic'][i] == 3 and new['SentimentHeadlineAbs'][i] == -1:
        s = [w for w in new['Headline'][i].lower().split() if not w in set(stopwords.words('english')) and not w in ['palestine','palestinian','israel','israeli','said','state','palestinians','ramallah','east','monday','west','un']]
        palestine_headline_neg += ' '+ ' '.join(s)




Counter0 = Counter(obama_title_pos.split())
obama_title_pos_most_occur = Counter0.most_common(24)

Counter1 = Counter(obama_title_neg.split())
obama_title_neg_most_occur = Counter1.most_common(24)

Counter2 = Counter(obama_headline_pos.split())
obama_headline_pos_most_occur = Counter2.most_common(24)

Counter3 = Counter(obama_headline_neg.split())
obama_headline_neg_most_occur = Counter3.most_common(24)



Counter00 = Counter(economy_title_pos.split())
economy_title_pos_most_occur = Counter00.most_common(24)

Counter10 = Counter(economy_title_neg.split())
economy_title_neg_most_occur = Counter10.most_common(24)

Counter20 = Counter(economy_headline_pos.split())
economy_headline_pos_most_occur = Counter20.most_common(24)

Counter30 = Counter(economy_headline_neg.split())
economy_headline_neg_most_occur = Counter30.most_common(24)



Counter000 = Counter(microsoft_title_pos.split())
microsoft_title_pos_most_occur = Counter000.most_common(24)

Counter100 = Counter(microsoft_title_neg.split())
microsoft_title_neg_most_occur = Counter100.most_common(24)

Counter200 = Counter(microsoft_headline_pos.split())
microsoft_headline_pos_most_occur = Counter200.most_common(24)

Counter300 = Counter(microsoft_headline_neg.split())
microsoft_headline_neg_most_occur = Counter300.most_common(24)



Counter0000 = Counter(palestine_title_pos.split())
palestine_title_pos_most_occur = Counter0000.most_common(24)

Counter1000 = Counter(palestine_title_neg.split())
palestine_title_neg_most_occur = Counter1000.most_common(24)

Counter2000 = Counter(palestine_headline_pos.split())
palestine_headline_pos_most_occur = Counter2000.most_common(24)

Counter3000 = Counter(palestine_headline_neg.split())
palestine_headline_neg_most_occur = Counter3000.most_common(24)


title_pos = obama_title_pos_most_occur+ economy_title_pos_most_occur+ palestine_title_pos_most_occur+ microsoft_title_pos_most_occur
title_neg =  obama_title_neg_most_occur+ economy_title_neg_most_occur+ palestine_title_neg_most_occur+ microsoft_title_neg_most_occur
headline_pos =  obama_headline_pos_most_occur+ economy_headline_pos_most_occur+ palestine_headline_pos_most_occur+ microsoft_headline_pos_most_occur
headline_neg =   obama_headline_neg_most_occur+ economy_headline_neg_most_occur+ palestine_headline_neg_most_occur+ microsoft_headline_neg_most_occur

title_pos = sorted(title_pos, key=lambda x: x[0])
title_neg = sorted(title_neg, key=lambda x: x[0])
headline_pos = sorted(headline_pos, key=lambda x: x[0])
headline_neg = sorted(headline_neg, key=lambda x: x[0])

'''



def preprocess(sentence):
    sentence = re.sub(r'[^\x00-\x7f]',r'', sentence.lower())
    sentence = sentence.translate(translator)
    #sentence = re.sub(r'[0-9]',r'',sentence)
    
    return sentence

## DATETIME FEATURE ENGINEERING

df_train['PublishDate'] = pd.to_datetime(df_train['PublishDate'], errors = 'coerce')
assert df_train.PublishDate.isnull().sum() == 0, 'missing'
## MON = 0 || FRI = 4
df_train['publishDay'] = df_train['PublishDate'].dt.dayofweek
## January = 1 || December = 12
df_train['publishMonth'] = df_train['PublishDate'].dt.month


## Categorical Encoding

Topics = sorted(['obama','economy','microsoft','palestine'])
df_train['Topic'] = df_train['Topic'].apply(lambda x: Topics.index(x))

## s1 = df_train.Source.to_list()
Source = sorted(df_train.Source.to_list())
Source = set(Source)
Source = list(Source)
Source.append('')
df_train['Source'] = df_train['Source'].apply(lambda x: Source.index(x))

## Pre Processing of Textual Data

df_train['Title'] = df_train['Title'].apply(preprocess)
df_train['Headline'] = df_train['Headline'].apply(preprocess)


df_train_title = df_train[[
 'Title',
 'Source',
 'Topic',
 'Facebook',
 'GooglePlus',
 'LinkedIn',
 'publishDay',
 'publishMonth',
 'SentimentTitle']].copy()

df_train_headline = df_train[[
 'Headline',
 'Source',
 'Topic',
 'Facebook',
 'GooglePlus',
 'LinkedIn',
 'publishDay',
 'publishMonth',
 'SentimentHeadline']].copy()

## Using Vader Sentiment Analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# New words and values  -4 to 4
new_words_Title = {
    'boost': 4,
    'big':3,
    'surface':2,
    'action':1,
    'aid':1,
    'gaza':2,
    'administration': 0.8,
    'deal': 0.8,
    'islamic': 0.8,
    'muslim': 0.8,
    'nominee': 0.8,
    'android':0.5,
    'leader':0.5,
    'cloud':-0.2,
    'control':-0.2,
    'cuba':-0.2,
    'edge':-0.2,
    'budget':-0.4,
    'business':-0.4,
    'amid':-1,
    'fight':-1,
    'final':-1,
    'russian':-1,
    'conflict': -2,
    'last': -2,
    'recession': -2,
    'low': -2,
    'isis': -0.5,
    'oil':-3,
    'shot':-3,
    'crisis': -4,
    'dead': -4,
    'killed': -4,
    'terror': -4,
    'war': -4,
    'weak': -4,
}

new_words_Headline = {
    'first':2,
    'lady': 2,
    'americans': 0.5,
    'abbas': 0.5,
    'mahmoud': 0.5,
    'liberation':0.2,
    'time':0.2,
    'congress':-1,
    'national':-0.2,
    'wafa':-0.2,
    'police':-0.8,
    'cuba':-1,
    'behind':-1,
    'report':-1,
    'conflict': -2,
    'final': -2,
    'rate': -2,
    'growth': -1,
    'visit': -1,
}

vaderTitle = SentimentIntensityAnalyzer()
vaderTitle.lexicon.update(new_words_Title)

vaderHeadline = SentimentIntensityAnalyzer()
vaderHeadline.lexicon.update(new_words_Headline)

def scoreTitle(s):
    vs1 = vaderTitle.polarity_scores(s)
    return vs1['compound']

def scoreHeadline(s):
    vs2 = vaderHeadline.polarity_scores(s)
    return vs2['compound']

df_train_title['VaderSentimentScore'] = df_train_title['Title'].apply(scoreTitle)
df_train_headline['VaderSentimentScore'] = df_train_headline['Headline'].apply(scoreHeadline)



from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

## 1st -> Title

df_title = df_train_title[['Source',
 'Topic',
 'Facebook',
 'GooglePlus',
 'LinkedIn',
 'publishDay',
 'publishMonth',
 'VaderSentimentScore',
 'SentimentTitle']].copy()


train_x, test_x, train_y, test_y = train_test_split(
        df_title.iloc[:,0:-1], df_title.iloc[:,-1:], test_size=0.2, random_state = 0
        )

'''
train_x = df_title.iloc[:,0:-1]
train_y = df_title.iloc[:,-1:]
'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)

train_xS = scaler.transform(train_x)
test_xS = scaler.transform(test_x)


pca =PCA(.95)
pca.fit(train_xS)

train_xS_pca = pca.transform(train_xS)
test_xS_pca = pca.transform(test_xS)

train_y = train_y.to_numpy()
test_y = test_y.to_numpy()

'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 8)
X_poly = poly_reg.fit_transform(train_xS_pca)
'''

import lightgbm as lgb

SEARCH_PARAMS = {'learning_rate': 0.4,
                'max_depth': 15,
                'feature_fraction': 0.8,
                'subsample': 0.2}

FIXED_PARAMS={'objective': 'regression',
             'metric': 'mae',
             'bagging_freq':5,
             #'boosting':'dart',
             'boosting':'gbdt',
             'num_boost_round':300,
             'early_stopping_rounds':30
             }

train_data = lgb.Dataset(train_xS_pca, label=train_y)
valid_data = lgb.Dataset(test_xS_pca, label=test_y, reference=train_data)

params = {'metric':FIXED_PARAMS['metric'],
          'num_leaves': 10,
 'objective':FIXED_PARAMS['objective'],
 **SEARCH_PARAMS, **FIXED_PARAMS}

model = lgb.train(params, train_data,                     
                     valid_sets=[valid_data],
                     #num_boost_round=FIXED_PARAMS['num_boost_round'],
                     #early_stopping_rounds=FIXED_PARAMS['early_stopping_rounds'],
                     valid_names=['valid']
                )

try:
    score = model.best_score['valid']['mae']
except:
    score = None
    
'''
model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
'''


'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_xS_pca, train_y)
'''

##################################################################
## 2nd -> Headline

df_headline = df_train_headline[['Source',
 'Topic',
 'Facebook',
 'GooglePlus',
 'LinkedIn',
 'publishDay',
 'publishMonth',
 'VaderSentimentScore',
 'SentimentHeadline']].copy()

train_x1, test_x1, train_y1, test_y1 = train_test_split(
        df_title.iloc[:,0:-1], df_title.iloc[:,-1:], test_size=0.2, random_state = 0
        )
'''
train_x1 = df_headline.iloc[:,0:-1]
train_y1 = df_headline.iloc[:,-1:]
'''
scaler1 = StandardScaler()
scaler1.fit(train_x1)

train_xS1 = scaler1.transform(train_x1)
test_xS1 = scaler1.transform(test_x1)

pca1 =PCA(.95)
pca1.fit(train_xS1)

train_xS_pca1 = pca1.transform(train_xS1)
test_xS_pca1 = pca1.transform(test_xS1)

train_y1 = train_y1.to_numpy()
test_y1 = test_y1.to_numpy()


train_data1 = lgb.Dataset(train_xS_pca1, label=train_y1)
valid_data1 = lgb.Dataset(test_xS_pca1, label=test_y1, reference=train_data1)

model1 = lgb.train(params, train_data1,                     
                     valid_sets=[valid_data1],
                     #num_boost_round=FIXED_PARAMS['num_boost_round'],
                     
                     #early_stopping_rounds=FIXED_PARAMS['early_stopping_rounds'],
                     valid_names=['valid']
                )

try:
    score1 = model1.best_score['valid']['mae']
except:
    score1 = None

'''
poly_reg1 = PolynomialFeatures(degree = 8)
X_poly1 = poly_reg1.fit_transform(train_xS_pca1)
'''

# ---------------

'''
regressor1 = LinearRegression()
regressor1.fit(train_xS_pca1, train_y1)
'''

################################################################################


## TEST SET
## DATETIME FEATURE ENGINEERING

#df_test.dropna(inplace = True)
#df_test.reset_index(inplace = True)

null_values = df_test.loc[pd.isnull(df_test).any(1), :].index.values
null_list = list(null_values)

df_test['PublishDate'] = pd.to_datetime(df_test['PublishDate'], errors = 'coerce')
assert df_test.PublishDate.isnull().sum() == 0, 'missing'

## MON = 0 || FRI = 4
df_test['publishDay'] = df_test['PublishDate'].dt.dayofweek
## January = 1 || December = 12
df_test['publishMonth'] = df_test['PublishDate'].dt.month


## Categorical Encoding

Topics = sorted(['obama','economy','microsoft','palestine'])
df_test['Topic'] = df_test['Topic'].apply(lambda x: Topics.index(x))

## Pre Processing of Textual Data

df_test['Title'] = df_test['Title'].apply(preprocess)
df_test['Headline'] = df_test['Headline'].apply(preprocess)

src = []
for i in trange(0,len(df_test)):
    try:
        src.append(Source.index(df_test['Source'][i]))
    except:
        src.append(Source.index(''))
df_test['Source'] = src

df_title_test = df_test[['Source',
 'Topic',
 'Facebook',
 'GooglePlus',
 'LinkedIn',
 'publishDay',
 'publishMonth']].copy()
df_title_test['VaderSentimentScore'] = df_test['Title'].apply(scoreTitle)


df_headline_test = df_test[['Source',
 'Topic',
 'Facebook',
 'GooglePlus',
 'LinkedIn',
 'publishDay',
 'publishMonth']].copy()
df_headline_test['VaderSentimentScore'] = df_test['Headline'].apply(scoreHeadline)


df_title_test = scaler.transform(df_title_test)
df_title_test = pca.transform(df_title_test)

df_headline_test = scaler1.transform(df_headline_test)
df_headline_test = pca1.transform(df_headline_test)

#title_pred = classifier.predict(df_title_test)
#headline_pred = classifier1.predict(df_headline_test)


title_pred = model.predict(df_title_test)
headline_pred = model1.predict(df_headline_test)

df = pd.DataFrame()
df['IDLink'] = df_test['IDLink']
df['SentimentTitle'] = title_pred
df['SentimentHeadline'] = headline_pred

#df.to_csv('submissions2.csv', index=False)
#df_all = pd.read_csv('submissions.csv')