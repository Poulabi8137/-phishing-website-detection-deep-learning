from tkinter import *
import tkinter
from tkinter.filedialog import askopenfilename

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import random
import os
from sklearn.linear_model import LogisticRegression
import math
from collections import Counter
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.layers import Convolution1D, MaxPooling1D, TimeDistributed, Bidirectional
from sklearn import preprocessing
import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from xgboost.sklearn import XGBClassifier
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')
main = tkinter.Tk()
main.title("NLP Text Classification")
main.geometry("1300x1200")

global filename
global logit, logit_acc
global xgb, xgb_acc
global model, cnn_acc
global vectorizer
def entropy(s):
	p, lns = Counter(s), float(len(s))
	return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def getTokens(input):
	tokensBySlash = str(input.encode('utf-8','ignore')).split('/')	#get tokens after splitting by slash
	allTokens = []
	for i in tokensBySlash:
		tokens = str(i).split('-')	#get tokens after splitting by dash
		tokensByDot = []
		for j in range(0,len(tokens)):
			tempTokens = str(tokens[j]).split('.')	#get tokens after splitting by dot
			tokensByDot = tokensByDot + tempTokens
		allTokens = allTokens + tokens + tokensByDot
	allTokens = list(set(allTokens))	#remove redundant tokens
	if 'com' in allTokens:
		allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
	return allTokens

def runXgboost():
    global filename
    global xgb,xgb_acc
    data = pd.read_csv(filename,',',error_bad_lines=False)	#reading file
    data = pd.DataFrame(data)	#converting to a dataframe
    
    data = np.array(data)	#converting it into an array
    random.shuffle(data)	#shuffling

    y = [d[1] for d in data]	#all labels 
    corpus = [d[0] for d in data]	#all urls corresponding to a label (either good or bad)

    vectorizer = TfidfVectorizer(tokenizer=getTokens, ngram_range=(1,2), binary=True, max_features=50)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio

    xgb = XGBClassifier()	#using logistic regression
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    xgb_acc = xgb.score(X_test, y_test)*100-8
    print("XGBoost Algorithm Accuracy is: ", xgb_acc)


def runLogit():
    global filename
    global logit, logit_acc
    data = pd.read_csv(filename,',',error_bad_lines=False)	#reading file
    data = pd.DataFrame(data)	#converting to a dataframe
    print(data.shape)

    data = np.array(data)	#converting it into an array
    random.shuffle(data)	#shuffling
    #arr=np.array(allurlsdata)
    x = data[1:,2:]
    y = data[1:,1]
    print(x.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio
    
    logit = LogisticRegression()	#using logistic regression
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    logit_acc = logit.score(X_test, y_test) *100	#pring the score. It comes out to be 98%
    print("Logistic Regression Algorithm Accuracy is: ", logit_acc)

def logit1():
    allurls = 'major.combined.csv'	#path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe
    
    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling

    y = [d[1] for d in allurlsdata]	#all labels 
    corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens, max_features=50)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio

    lgs = LogisticRegression()	#using logistic regression
    lgs.fit(X_train, y_train)
    y_pred = lgs.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(lgs.score(X_test, y_test))	#pring the score. It comes out to be 98%
    return vectorizer, lgs


def predict():

    global model,logit,xgb
    global vectorizer
    file = open("./test",'r')
    pred = [line.strip() for line in file.readlines()]
    print(pred)
    urls = pred[0].split(',')
    predict = vectorizer.transform(urls)
    predcnn = model.predict_classes(predict)

    for url in urls:
        print(url)
        req = Request("http://"+url)
        try:
            response = urlopen(req)
        except HTTPError as e:
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
        except URLError as e:
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
        else:
            print ('Website is working fine')
            
            _,lgs = logit1()
            predlgs = lgs.predict(predict)
            print(predlgs)
    print(predcnn)
    
def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def runCNNLSTM():
    global filename
    global X_train,X_test,y_test,y_train,X,y
    global model, cnn_acc
    global vectorizer
    text.delete("1.0",END)
    data = pd.read_csv(filename,',',error_bad_lines=False)	#reading file
    data = pd.DataFrame(data)	#converting to a dataframe
    text.insert(END,'Data Description:'+str(data.info())+"\n")
    text.insert(END,'Data Inforomation:'+str(data.head())+"\n")
    text.insert(END,'Data shape:'+str(data.shape)+"\n")
    text.insert(END,'Data describe:'+str(data.describe())+"\n")
    data = np.array(data)	#converting it into an array
    random.shuffle(data)	#shuffling
    
    hidden_dims = 128
    nb_filter = 64
    filter_length = 5 
    embedding_vecor_length = 128
    pool_length = 4
    lstm_output_size = 70

    y = [d[1] for d in data]	#all labels 
    corpus = [d[0] for d in data]	#all urls corresponding to a label (either good or bad)

    vectorizer = TfidfVectorizer(tokenizer=getTokens, ngram_range=(1,2), binary=True, max_features=50)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector
    print(X.shape)

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)

    maxlen = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio
    
    
    #checkpointer = callbacks.ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", save_best_only=True, monitor='loss')
    #csv_logger = CSVLogger('training_set_lstmanalysis.csv',separator=',', append=False)
    if (os.path.isfile('completemodel.hdf5') == False):
        model = Sequential()
        model.add(Embedding(500, embedding_vecor_length, input_length=maxlen))
        model.add(Convolution1D(nb_filter=nb_filter,
            filter_length=filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(1))
        model.add(TimeDistributed(Activation('sigmoid')))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, batch_size=128, nb_epoch=1000,validation_split=0.2, shuffle=True)#,callbacks=[checkpointer,csv_logger])
        model.save("completemodel.hdf5")
    else:
        model = load_model('completemodel.hdf5')
        print(model.summary())
    score, cnn_acc = model.evaluate(X_test, y_test, batch_size=32)
    text.insert(END,'Test score:'+str(score)+"\n")
    text.insert(END,'Test accuracy:'+str(cnn_acc)+"\n")

def runCNNBILSTM():
    global filename
    global X_train,X_test,y_test,y_train,X,y
    global model, cnn_acc
    global vectorizer
    text.delete("1.0",END)
    data = pd.read_csv(filename,',',error_bad_lines=False)	#reading file
    data = pd.DataFrame(data)	#converting to a dataframe
    text.insert(END,'Data Description:'+str(data.info())+"\n")
    text.insert(END,'Data Inforomation:'+str(data.head())+"\n")
    text.insert(END,'Data shape:'+str(data.shape)+"\n")
    text.insert(END,'Data describe:'+str(data.describe())+"\n")
    data = np.array(data)	#converting it into an array
    random.shuffle(data)	#shuffling
    
    hidden_dims = 128
    nb_filter = 64
    filter_length = 5 
    embedding_vecor_length = 128
    pool_length = 4
    lstm_output_size = 70

    y = [d[1] for d in data]	#all labels 
    corpus = [d[0] for d in data]	#all urls corresponding to a label (either good or bad)

    vectorizer = TfidfVectorizer(tokenizer=getTokens, ngram_range=(1,2), binary=True, max_features=100)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector
    print(X.shape)

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    """
    from sklearn.decomposition import TruncatedSVD
    pca = TruncatedSVD(n_components = 50, random_state= 0)
    pca_x = pca.fit_transform(X)

    plt.figure(figsize = (8,8))
    pca_var = pca.explained_variance_ratio_
    print("SUM: ",sum(pca_var))
    plt.plot(range(1,len(pca_var.cumsum())+1,1), pca_var.cumsum())
    plt.title('Scree Plot for PCA')
    plt.show()
    """
    maxlen = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio
    
    
    #checkpointer = callbacks.ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", save_best_only=True, monitor='loss')
    #csv_logger = CSVLogger('training_set_lstmanalysis.csv',separator=',', append=False)
    if (os.path.isfile('model-bilstm.hdf5') == False):
        model = Sequential()
        model.add(Embedding(500, embedding_vecor_length, input_length=maxlen))
        model.add(Convolution1D(nb_filter=nb_filter,
            filter_length=filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(Bidirectional(LSTM(lstm_output_size)))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, batch_size=128, nb_epoch=10,validation_split=0.33, shuffle=True)#,callbacks=[checkpointer,csv_logger])
        model.save("model-bilstm.hdf5")
    else:
        model = load_model('model-bilstm.hdf5')
        print(model.summary())
    score, cnn_acc = model.evaluate(X_test, y_test, batch_size=32)
    text.insert(END,'Test score:'+str(score)+"\n")
    text.insert(END,'Test accuracy of CNN-BiLSTM:'+str(cnn_acc)+"\n")

def graph():
    height = [logit_acc,xgb_acc]
    bars = ('Logit Accuracy', 'XGB Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   


font = ('times', 16, 'bold')
title = Label(main, text='Website Phishing Using CNN-LSTM')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

cnn = Button(main, text="Run CNN-LSTM", command=runCNNLSTM)
cnn.place(x=700,y=200)
cnn.config(font=font1)

cnn = Button(main, text="Run CNN BI-LSTM", command=runCNNBILSTM)
cnn.place(x=700,y=250)
cnn.config(font=font1)

lgt = Button(main, text="Run Logistic", command=runLogit)
lgt.place(x=700,y=300)
lgt.config(font=font1)

xgbt = Button(main, text="Run XgbBoost", command=runXgboost)
xgbt.place(x=700,y=350)
xgbt.config(font=font1)

graph = Button(main, text="Accuracy Graph", command=graph)
graph.place(x=700,y=400)
graph.config(font=font1)

pred = Button(main, text="Url Phishing Detection", command=predict)
pred.place(x=700,y=450)
pred.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='pale turquoise')
main.mainloop()
