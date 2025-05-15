import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Przetwarza dane do formatu numerycznego i zwraca X i y
def preprocess_data(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y