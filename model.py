from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer as ps
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import joblib
import csv
## Load DataSet
raw_data=pd.read_csv('train.csv')
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
# !raw_data is clean
## Generate Sparse Matrix
c_vect=CountVectorizer(max_features=2700)
c_vect_x=c_vect.fit_transform(raw_data['text'])
c_vect_arr=c_vect_x.toarray()
## Done
y=raw_data['sentiment'].values
print("Data Preprocessing Complete..Model Training in Progress")
## Start Building Model
model=DecisionTreeClassifier()
model.fit(c_vect_arr,y)
print("Model training Complete")
print(model.score(c_vect_arr,y))
print("Dumping Model")
joblib_dmp="model.pkl"
joblib.dump(model,joblib_dmp)
file=open('sparse_feat.csv','w',newline='');
writer=csv.writer(file)
writer.writerow(c_vect.get_feature_names())
file.flush()
file.close()
print("Done")
