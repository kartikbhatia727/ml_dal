#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import xml.etree.ElementTree as et 

xtree = et.parse("429411newsML.xml")
xroot = xtree.getroot() 

df_cols = [ "headline","text","bip:topics","dc.date.published","itemid","XMLfilename"]
rows = []

print(xroot)


# In[2]:


xroot.attrib.get("itemid")


# In[115]:


for u in xroot.iter('dc'):
    if u.get('element') =='dc.date.published':
        print(u.get('value'))


# In[8]:


import pandas as pd 
import xml.etree.ElementTree as et 

xtree = et.parse("429411newsML.xml")
xroot = xtree.getroot() 
for t in xroot.iter('codes'):
    if t.get('class') =='bip:topics:1.0':
        for i in t:
            print(i.get('code'))


# In[18]:


directory = os.path.abspath('')

for filename in os.listdir(directory):
    if filename.endswith('.xml'):
        print(filename)


# In[2]:


import os


# In[6]:


os.path.dirname('/Users/kartik/Downloads/Machine learning/Data/Data/19970310')


# In[22]:


os.path.abspath('/Users/kartik/Downloads/Machine learning/Data/Data/19970310')


# In[30]:


import pandas as pd 
import xml.etree.ElementTree as et

directory = os.path.abspath('/Users/kartik/Downloads/Machine learning/Data/Data/19970310')
lst=[]
for filename in os.listdir(directory):
    if filename.endswith('.xml'):
        fullname = os.path.join(directory, filename)
        tree = et.parse(fullname)
        root = tree.getroot() 
        for t in root.iter('codes'):
            if t.get('class') =='bip:topics:1.0':
                for i in t:
                    lst.append(i.get('code'))
lst1=set(lst)                    
print(lst1)


# In[10]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pandas as pd 
import nltk
import xml.etree.ElementTree as et 


def filtered_sentence (filename):
    xtree = et.parse(filename)
    xroot = xtree.getroot() 

    lst=[]

    for i in xroot:
        if i.tag=='text':
            for j in i:
                lst.append(j.text)
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = " " 
    for ls in lst:
        word_tokens = word_tokenize(ls)
        lst1=[]
        for w in word_tokens: 
            if w not in stop_words: 
                lst1.append(w)
        test=' '.join(lst1) 
        filtered_sentence= filtered_sentence +' '+ test
    return filtered_sentence

filtered_sentence("429411newsML.xml")


# In[10]:



from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
example_sent = "Council Regulation (EC) No 390/97 of 20 December 1996 fixing"
  
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(example_sent)
print(word_tokens)

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)


print(filtered_sentence)


# In[2]:


import os 
import xml.etree.cElementTree as et
import pandas as pd
 
def extract_df_xml():
    directory = os.path.abspath('/Users/kartik/Downloads/Machine learning/Data/Data/19970310')
    rows=[]
    for filename in os.listdir(directory):
        if filename.endswith('.xml'): 
            fullname = os.path.join(directory, filename)
            tree = et.parse(fullname)
            xroot=tree.getroot()
            df_cols = [ "headline","text","bip:topics","dc.date.published","itemid","XMLfilename"]
            lst=[]
            lst1=[]
            itemid=xroot.attrib.get("itemid")
            for node in xroot:
                XMLfilename=filename
                if node.tag =='headline':
                    headline=node.text
                if node.tag=='text':
                    for j in node:
                        lst.append(j.text)
                    text=' '.join(lst)
            for tags in xroot.iter('dc'):
                if tags.get('element') =='dc.date.published':
                    dc=tags.get('value')
            for codes in xroot.iter('codes'):
                if codes.get('class') =='bip:topics:1.0':
                    for code in codes:
                        lst1.append(code.get('code'))   
                    bip=lst1
        
            rows.append({"headline":headline, "text":text, "bip:topics":bip,"dc.date.published":dc,"itemid":itemid,"XMLfilename":XMLfilename})
            df_xml = pd.DataFrame(rows,columns=df_cols)
            
    return df_xml

extract_df_xml()


# In[3]:


import nltk
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
from nltk.corpus import stopwords 
from textblob import TextBlob 
from nltk.tokenize import word_tokenize


def feature_df(df_old):
    rows=[]
    index=0
    for i in df_old['text']:
        df_cols = [ "lemmatized_output","keywords","polarity","bip:topics"]
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english')) 
        filtered_sentence = " " 
        word_tokens = word_tokenize(i)
        lst1=[]
        for w in word_tokens: 
            if w not in stop_words: 
                lst1.append(w)
        test=' '.join(lst1) 
        filtered_sentence= filtered_sentence +' '+ test
        word_list = nltk.word_tokenize(filtered_sentence)
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
        r = Rake(min_length=2, max_length=4)
        r.extract_keywords_from_text(filtered_sentence)
        keywords=', '.join(r.get_ranked_phrases())
        blob = TextBlob(filtered_sentence)
        polarity=blob.sentiment.polarity
        bip=df_old['bip:topics'][index][0]
        rows.append({"lemmatized_output":lemmatized_output, "keywords":keywords, "polarity":polarity,"bip:topics":bip})
        df_new = pd.DataFrame(rows,columns=df_cols)
        index+=1
    return df_new  

#odf=extract_df_xml()
#print(feature_df(odf))


# In[11]:


from rake_nltk import Rake

r = Rake()
text = """Google quietly rolled out a new way for Android users to listen 
to podcasts and subscribe to shows they like, and it already works on 
your phone. Podcast production company Pacific Content got the exclusive 
on it.This text is taken from Google news."""

r.extract_keywords_from_text(text)

b=r.get_ranked_phrases()
print(', '.join(b))


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import preprocessing

odf=extract_df_xml()
df=feature_df(odf)
le = preprocessing.LabelEncoder()
df['bip:topics'] = le.fit_transform(df['bip:topics'])
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.7,lowercase=True,stop_words=stopwords.words('english'),ngram_range = (1,1),tokenizer = token.tokenize)
X= cv.fit_transform(df['lemmatized_output'])
y = df.pop('bip:topics')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print ("Score:", model.score(X_test, y_test))


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np 


odf=extract_df_xml()
df=feature_df(odf)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.7,lowercase=True,stop_words=stopwords.words('english'),ngram_range = (1,1),tokenizer = token.tokenize)
X= cv.fit_transform(df['lemmatized_output'])

y = df.pop('bip:topics')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
model= classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.640074211502783
#Recall:  0.6673114119922631
#Precision 0.6973233305948217
#F1 Score: 0.5987629523878211


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np 


odf=extract_df_xml()
df=feature_df(odf)

classifier=train_model(RandomForestClassifier(n_estimators=100, random_state=0),df)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.640074211502783
#Recall:  0.6673114119922631
#Precision 0.6973233305948217
#F1 Score: 0.5987629523878211


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np

odf=extract_df_xml()
df=feature_df(odf)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.7,lowercase=True,stop_words=stopwords.words('english'),ngram_range = (1,1),tokenizer = token.tokenize)
X= cv.fit_transform(df['lemmatized_output'])
y = df.pop('bip:topics')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

model = MultinomialNB().fit(X_train, y_train)
y_pred= model.predict(X_test)

print ("Score:", model.score(X_test, y_test))
print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print(metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print(metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print(metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))


# In[23]:


from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

odf=extract_df_xml()
df=feature_df(odf)
model= train_model(DecisionTreeClassifier(random_state=0),df)
y_pred= model.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.4842300556586271
#Recall:  0.4869402985074627
#Precision 0.5012242303317657
#F1 Score: 0.48514006036355933


# In[19]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics

odf=extract_df_xml()
df=feature_df(odf)

classifier=train_model(svm.SVC(kernel='linear',gamma='scale'),df)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.6938775510204082
#Recall:  0.7030075187969925
#Precision 0.7082971815163361
#F1 Score: 0.6968338667107297


# In[29]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics

odf=extract_df_xml()
df=feature_df(odf)

classifier=train_model(svm.SVC(kernel='linear',C=0.1,gamma='scale'),df)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.6938775510204082
#Recall:  0.7030075187969925
#Precision 0.7082971815163361
#F1 Score: 0.6968338667107297


# In[30]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics

odf=extract_df_xml()
df=feature_df(odf)

classifier=train_model(svm.SVC(kernel='linear',C=0.1,gamma='scale'),df)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.6938775510204082
#Recall:  0.7030075187969925
#Precision 0.7082971815163361
#F1 Score: 0.6968338667107297


# In[26]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics

odf=extract_df_xml()
df=feature_df(odf)

classifier=train_model(svm.LinearSVC(),df)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
#Accuracy :  0.6938775510204082
#Recall:  0.7030075187969925
#Precision 0.7082971815163361
#F1 Score: 0.6968338667107297


# In[18]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics

odf=extract_df_xml()
df=feature_df(odf)

classifier= train_model(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1),df)
y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def train_model(classifier,dataframe):
    df=dataframe

    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.7,lowercase=True,stop_words=stopwords.words('english'),ngram_range = (1,1),tokenizer = token.tokenize)
    X= cv.fit_transform(df['lemmatized_output'])
    y = df.pop('bip:topics')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

    classifier.fit(X_train, y_train)

    return classifier

odf=extract_df_xml()
df=feature_df(odf)
m=train_model(MultinomialNB(),df)
print(y_test)
print(m)


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics 


def quality_of_model(classifier,dataframe):
    df=dataframe
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.7,lowercase=True,stop_words=stopwords.words('english'),ngram_range = (1,1),tokenizer = token.tokenize)
    X= cv.fit_transform(df['lemmatized_output'])
    y = df.pop('bip:topics')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

    model = classifier.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    
    print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_pred)))
    print(metrics.confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))
    print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
    print("Recall: ",metrics.recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    print("Precision",metrics.precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    print("F1 Score:",metrics.f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    

odf=extract_df_xml()
df=feature_df(odf)
quality_of_model(MultinomialNB(),df)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import metrics

odf=extract_df_xml()
df=feature_df(odf)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.7,lowercase=True,stop_words=stopwords.words('english'),ngram_range = (1,1),tokenizer = token.tokenize)
X= cv.fit_transform(df['lemmatized_output'])
y = df.pop('bip:topics')

kf = KFold(n_splits=10, random_state=42, shuffle=False)

lm= MultinomialNB()

results = cross_val_score(lm,X,y,cv=kf)
print("Accuracy: %.2f%%" % (results.mean()*100.0)) 
y_pred = cross_val_predict(lm, X, y, cv=kf)
print(accuracy_score(y,y_pred))
print(metrics.classification_report(y,y_pred,labels=np.unique(y_pred)))
print(metrics.confusion_matrix(y, y_pred,labels=np.unique(y_pred)))
print(metrics.recall_score(y, y_pred,average="weighted",labels=np.unique(y_pred)))
print(metrics.precision_score(y, y_pred,average="weighted",labels=np.unique(y_pred)))
print(metrics.f1_score(y, y_pred,average="weighted",labels=np.unique(y_pred)))

