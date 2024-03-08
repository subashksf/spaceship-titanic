import config
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_boolean(_data, BOOL_FEATURE):
    _data[BOOL_FEATURE] = _data[BOOL_FEATURE].apply(lambda x: 1 if x == True else 0)
    return _data

def label_encode(_data, TO_ENCODE_FEATURE):
    label_encoder = LabelEncoder()
    _data[TO_ENCODE_FEATURE] = label_encoder.fit_transform(_data[TO_ENCODE_FEATURE])
    
    return _data

def preprocess_cabin(_data):
    cabin_data = _data['Cabin'].str.split("/", n=2, expand=True)

    _data['deck'] = cabin_data[0]
    _data['num'] = cabin_data[1]
    _data['side'] = cabin_data[2]

    return _data

def preprocess_age(_data):
    bins= [0,10,20,30,40,50,60,70,80,90]
    labels = ['1s','10s','20s','30s','40s','50s','60s','70s','80s']

    _data['AgeGroup'] = pd.cut(_data['Age'], bins=bins, labels=labels, right=False)

    return _data
