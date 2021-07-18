
import pandas as pd 
import numpy as np
#from numpy import array
from numpy import asarray
from numpy import zeros
import xml.etree.ElementTree as ET 
import re
import os
import nltk
import string
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
#from sklearn.base import BaseEstimator
#from sklearn import utils as skl_utils
from sklearn.decomposition import PCA
#plt.style.use('ggplot')
import multiprocessing
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Flatten,Input
from keras import layers,utils 
from keras.layers.convolutional import Conv1D,MaxPooling1D,AveragePooling1D
from keras.layers.merge import concatenate
from sklearn.metrics import silhouette_score

#plt.style.use('fivethirtyeight')
#pd.set_option('display.max_columns', None)  
#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('max_colwidth', -1)
df_cols = ["headline","text","biptopic","biptopics","itemid","datepublished","filename"]
np.random.seed(500)

def xmlToDF (fullname,filename):
    xtree = ET.parse(fullname)
    xroot = xtree.getroot() 
    rows = []
    biptopics =[]
    itemid=xroot.attrib.get ("itemid")
    for elem in xtree.iterfind('headline'):
        headline = elem.text
        output_lst = [ET.tostring(child, encoding="utf8") for child in xroot.find('text')]  
    text= re.sub("<.*?>", "", ''.join(output_lst))
            
    for elem in xtree.iterfind('.//dc[@element="dc.date.published"]'):
        datepublished = elem.attrib.get('value')
    
    #biptopics : contain list of all biptopics in a xml file to answer point 4
    #biptopic : contain only first biptopic which i used in all further classification
    for elem in xtree.iterfind('.//codes[@class="bip:topics:1.0"]/code'):
        biptopics.append (elem.attrib.get('code'))
    
    if len(biptopics) > 0 : # to discard those xml files which dont have label i.e. biptopic
        rows.append ({"headline": headline,"text": text,"biptopic": biptopics[0],"biptopics":biptopics,
              "itemid": itemid,"datepublished":datepublished,
              "filename":filename})
    out_df = pd.DataFrame(rows, columns = df_cols)
    out_df['text'].replace(r'\s+|\\n', ' ', regex=True, inplace=True) 
    return out_df

def dataPreProcessing (df):
    stopwordlist = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    df['cleantext'] = [row.lower().strip() for row in df['text']]
    
    df['cleantext'] = [row.strip(string.punctuation) for row in df['cleantext']]
    
    df['cleantext'] = df['cleantext'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwordlist]))
    df['cleantext'] = df['cleantext'].apply(lambda x: ' '.join([lemmatizer.lemmatize(item) for item in x.split()]))
    df['cleantext'] = df['cleantext'].apply(lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
           
    tv = TfidfVectorizer(max_df=0.9,min_df=0.1)
    tfidfVector = tv.fit_transform(df.cleantext)
    #print(tfidfVector.shape)
    model = KMeans(n_clusters = 24,init='k-means++',random_state=99,max_iter=100,n_init=20, n_jobs=-1)
    model.fit(tfidfVector)
    clusters = model.labels_.tolist()
    df['clusterID']=clusters
    #print(df['clusterID'].value_counts()) 
    terms = tv.get_feature_names()
    df.to_csv(r'/users/grad/akhtar/subdata/123.csv')
    return df,tfidfVector


def CNN (new_df):
    x = new_df['cleantext'].values
    y = new_df['biptopic'].values
    num_class = len(np.unique(y))
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25,random_state=1000)
    encoder = LabelEncoder()
    encoder.fit(y)
    entrainY = encoder.transform(y_train)
    entestY = encoder.transform(y_test)
    entrainY = utils.to_categorical(entrainY, num_class)
    entestY = utils.to_categorical(entestY, num_class)
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(x_train)
    X_train = tokenizer.texts_to_sequences(x_train)
    X_test = tokenizer.texts_to_sequences(x_test)
    vocab_size = len(tokenizer.word_index) + 1    
    maxlen = 500
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    embedding_dim = 50
    embeddings_index = dict()
    f = open('/users/grad/akhtar/subdata/glove.6B.50d.txt')
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = zeros((vocab_size, 50))
    for word, i in tokenizer.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    inputs = Input(shape=(maxlen,))
    model = Sequential()
    embedding=Embedding(vocab_size, embedding_dim,  weights=[embedding_matrix],input_length=maxlen,trainable=True)(inputs)
    conv1 = Conv1D(filters=100, kernel_size=5, activation='relu')(embedding)
    pool1 = AveragePooling1D(pool_size=2)(conv1)
    
    conv2 = Conv1D(filters=100, kernel_size=5, activation='relu')(pool1)
    pool2 = AveragePooling1D(pool_size=2)(conv2)
    
    conv3 = Conv1D(filters=100, kernel_size=5, activation='relu')(pool2)
    pool3 = AveragePooling1D(pool_size=2)(conv3)
    
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    merged = concatenate([flat1, flat2, flat3])
    dense1 = Dense(100, activation='relu')(merged)
    dense2 = Dropout(0.5)(dense1)
    outputs = Dense(num_class, activation='softmax')(dense2)
    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, entrainY,epochs=10,batch_size=50, verbose=0)
    loss, accuracy = model.evaluate(X_test, entestY, verbose=0)
    print(" Accuracy:  {:.4f}".format(accuracy*100))
    #### add here accuracy with metrics module

def silhouetteScore (data,max_k):
    iters = range(2, max_k+1, 2)
    for n_clusters in iters :
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=200, n_init=1)
        preds = clusterer.fit_predict(data)
        centers = clusterer.cluster_centers_
        score = silhouette_score (data, preds, metric='euclidean')
        print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

def silhouettePlot (data):
    model = KMeans(n_clusters = 24,init='k-means++',random_state=99,max_iter=100,n_init=20, n_jobs=-1)
    y_pred=model.fit_predict(data)
    score = silhouette_score(data, y_pred, metric='euclidean')
    print("Silhouette Score:",score)
    sklearn_pca = PCA(n_components = 2)
    reduced_features = sklearn_pca.fit_transform(data.toarray())
    fitted = model.fit(reduced_features)
    prediction = model.predict(reduced_features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=prediction, s=50, cmap='viridis')
    center = model.cluster_centers_
    plt.scatter(center[:, 0], center[:, 1], c='black', s=300, alpha=0.6)
    plt.savefig('/users/grad/akhtar/subdata/silhouetteKmeans.png')

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f,ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    ax.figure.savefig('/users/grad/akhtar/subdata/KMeans.png')
    
  
final_df = pd.DataFrame(data = None, columns= df_cols)
rootpath = '/users/grad/akhtar/Assg1Data'

for root, dirs, files in os.walk(rootpath):
        for filename in files:
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(root, filename)
            final_df=final_df.append (xmlToDF (fullname,filename))
            print(filename)
print("XMLs Uploaded in Dataframe Successfully")
final_df.index = range(len(final_df))
print( final_df.shape)
final_df,tfidfVector = dataPreProcessing (final_df)

#find_optimal_clusters(tfidfVector,50)
silhouettePlot (tfidfVector)
for i in range(0,24):
    new_df = final_df[final_df['clusterID'] == i]
    print("Cluster ID : {}".format(i))
    CNN(new_df)

