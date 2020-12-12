#!/usr/bin/env python
# coding: utf-8

# ### Import Necessary Libraries



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer , WordNetLemmatizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

import pickle
import warnings
warnings.filterwarnings('ignore')


# ### Import Dataset [Train , Test, ]

### Import Train dataset
train_df = pd.read_csv('train.tsv',sep='\t')  
train_df.head(5)


### Import Test dataset
test_df = pd.read_csv('test.tsv',sep='\t')
test_df.head(5)

### Import another dataset
dev_df = pd.read_csv('dev.tsv',sep='\t')
dev_df.head(5)


# ### Train Dataset Analysis & Preprocessing



### Data shape
train_df.shape

### Show Data Info
train_df.info()


### Describe Dtaset
train_df.describe()



### Data Label count
train_df['label'].value_counts()



### Show Plot for Train Dataset
plt.style.use('fivethirtyeight')
sns.countplot(data=train_df,x='label')


### Drop Duplicate Value
train_df.drop_duplicates(inplace = True)
train_df


train_df.describe()


plt.style.use('fivethirtyeight')
sns.countplot(data=train_df,x='label')


train_df['label'].value_counts()

train_df.isnull().sum()


# ### Test Dataset ANalysis and Preprocessing




### Show Test dataset shape
test_df.shape

### Show Test dataset Info
test_df.info()


### Describe Test dataset
test_df.describe()


### Counts Label of Test Dataset 
test_df['label'].value_counts()

### Show Label Plot for test dataset
plt.style.use('fivethirtyeight')
sns.countplot(data=test_df,x='label')


### Drop duplicate Rows
test_df.drop_duplicates(inplace = True)


test_df


test_df.describe()


### Show Label plot after drop duplicate rows
plt.style.use('fivethirtyeight')
sns.countplot(data=test_df,x='label')


test_df['label'].value_counts()



### Check Null Value
test_df.isnull().sum()


# ### Another Dataset ANalysis and Preprocessing


### Data shape
dev_df.shape


### Show Data Info
dev_df.info()


### Describe Dtaset
dev_df.describe()


# In[206]:


### Data Label count
dev_df['label'].value_counts()



### Show Plot for Train Dataset
plt.style.use('fivethirtyeight')
sns.countplot(data=dev_df,x='label')

### Drop duplicate Rows
dev_df.drop_duplicates(inplace = True)

dev_df

dev_df.describe()


### Show Label plot after drop duplicate rows
plt.style.use('fivethirtyeight')
sns.countplot(data=dev_df,x='label')

dev_df['label'].value_counts()


dev_df.isnull().sum()


# ### Indentify Train , Test and Another dataet value

X_train=train_df['text_a'].values
Y_train=train_df['label'].values


X_test=test_df['text_a'].values
Y_test=test_df['label'].values



X_dev=dev_df['text_a'].values
Y_dev=dev_df['label'].values


# ### Analysis & Preprocessing Train and Test  Dataset


(X_train.shape,Y_train.shape),(X_test.shape,Y_test.shape)


train_df.iloc[:,1].describe()


test_df.iloc[:,1].describe()

X_train_len=[len(str(i).split()) for i in X_train]
plt.hist(X_train_len)


X_test_len=[len(str(i).split()) for i in X_test]
plt.hist(X_test_len)


vocab_size=30000 
embedding_dimension=64 
turnc='post'#preprocessing step for pad_sequences
oov_tok='<OOV>'#oov stands for out of vocabulary tokens


# vectorising the text
vect = CountVectorizer(stop_words=None)


vect.fit(X_train)


vect.vocabulary_


vect.get_feature_names()

# transform
X_train_transformed = vect.transform(X_train)
X_test_tranformed =vect.transform(X_test)


print(X_test[:1])


print(X_test_tranformed)


from sklearn.naive_bayes import BernoulliNB

# Instantiate bernoulli NB object
bnb = BernoulliNB()

# Fit The 
bnb.fit(X_train_transformed,Y_train)

# predict class
y_pred_class = bnb.predict(X_test_tranformed)

# Predict probability
y_pred_proba =bnb.predict_proba(X_test_tranformed)


bnb


cv = CountVectorizer()


x_train = cv.fit_transform(X_train)
x_test = cv.fit_transform(X_test)
x_dev = cv.fit_transform(X_dev)




from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()



### Train Dataset Accuracy
model.fit(x_train,Y_train)
model.score(x_train,Y_train)



### Test Dataset Accuracy
model.fit(x_test,Y_test)
model.score(x_test,Y_test)



### Another Dataset Accuracy
model.fit(x_dev,Y_dev)
model.score(x_dev,Y_dev)


metrics.confusion_matrix(Y_test, y_pred_class)



confusion = metrics.confusion_matrix(Y_test, y_pred_class)
print(confusion)
#[row, column]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]



sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)

specificity = TN / float(TN + FP)

print("specificity",specificity)



precision = TP / float(TP + FP)

print("precision",precision)
print(metrics.precision_score(Y_test, y_pred_class))


print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(Y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(Y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(Y_test, y_pred_class))



y_pred_proba


from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


print (roc_auc)



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


# ### Check Model With Predict

review = ['机器背面似乎被撕了张什么标签，残胶还在。但是又看不出是什么标签不见了，该有的都在，怪','地理位置佳，在市中心。酒店服务好、早餐品种丰富。我住的商务数码房电脑宽带速度满意,房间还算干净，离湖南路小吃街近。']



cv_review = cv.transform(review)

model.predict(cv_review)


# ### Generate Pickle File


#Saving model
pickle.dump(model, open('saici_task.pkl', 'wb'))


#Testing model by loading it first
model1= pickle.load(open('saici_task.pkl', 'rb'))







# ### Thank you Saici.ai Team 





