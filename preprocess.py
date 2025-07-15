# preprocess.py
from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df, cols):
    for c in cols:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df

def extract_dep_hour(df):
    df['DepHour'] = df['CRSDepTime'].astype(str).str.zfill(4).str[:2].astype(int)
    return df