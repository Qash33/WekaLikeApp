import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(df, enc=None, scaler=None, fit=True):
    df = df.copy()
    excluded_cols = ['id', 'price', 'year', 'month']
    feature_columns = [col for col in df.columns if col not in excluded_cols]

    cat_features = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()

    # Imputacja
    if num_features:
        num_imputer = SimpleImputer(strategy='median')
        X_num = num_imputer.fit_transform(df[num_features])
    else:
        X_num = np.empty((len(df), 0))

    if cat_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat_imp = pd.DataFrame(cat_imputer.fit_transform(df[cat_features]), columns=cat_features)
        if fit or enc is None:
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_cat = enc.fit_transform(X_cat_imp)
        else:
            X_cat = enc.transform(X_cat_imp)
    else:
        X_cat = np.empty((len(df), 0))
        enc = None

    if X_num.shape[1] > 0:
        if fit or scaler is None:
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X_num)
        else:
            X_num = scaler.transform(X_num)
    else:
        scaler = None

    X = np.hstack((X_cat, X_num))
    y = np.log(df['price'].values)
    feature_cols = cat_features + num_features

    return X, y, feature_cols, enc, scaler