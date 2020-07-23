from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk.stem import PorterStemmer as ps
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import joblib
import csv
## Load Model
model=joblib.load('model.pkl')
feats=list(csv.reader(open('sparse_feat.csv')))
##Load Test csv
raw_data=pd.read_csv('test.csv')
## Label Encoding
label_encoder=preprocessing.LabelEncoder()
raw_data['sentiment']=label_encoder.fit_transform(raw_data['sentiment'])
## Start Text PreProcess
stm=ps() # Protor Stemmer
stp=stopwords.words('english')
## Remove empty strings
nanval=float("NaN")
raw_data.replace("",nanval,inplace=True)
raw_data=raw_data.dropna()
raw_data=raw_data.reset_index(drop=True)
## Remove URLS
for i in range(len(raw_data['text'])):
    raw_data['text'][i]=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', raw_data['text'][i])
for i in range(len(raw_data['text'])):
    raw_data['text'][i]=re.sub('[^a-zA-Z]',' ',raw_data['text'][i]).lower()
    tmp2=""
    for j in raw_data['text'][i].split():
        if stm.stem(j) not in stp:
            tmp2+=stm.stem(j)
            tmp2+=' '
    raw_data['text'][i]=tmp2
## Remove empty strings
nanval=float("NaN")
raw_data.replace("",nanval,inplace=True)
raw_data=raw_data.dropna()
raw_data=raw_data.reset_index(drop=True)
## Done Preprocessing
## Generate Sparse Matrix
X=[]
y=raw_data['sentiment'].values
for i in raw_data['text']:
    ft=[]
    for j in feats[0]:
        if j not in i.split():
            ft.append(0)
        else:
            ft.append(1)
    X.append(ft)
    del(ft)
print(model.score(X,y))

