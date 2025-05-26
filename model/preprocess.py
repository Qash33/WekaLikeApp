import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Przetwarza dane do formatu numerycznego i zwraca X i y
def preprocess_data(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'], inplace=True)

    required_columns = ['price', 'squareMeters', 'rooms', 'city', 'buildYear', 'type']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolumna {col} jest wymagana w zbiorze danych!")

    # Kodowanie kolumn kategorycznych
    df['city'] = LabelEncoder().fit_transform(df['city'])
    df['type'] = LabelEncoder().fit_transform(df['type'])  # Dodano kodowanie dla 'type'

    # Zastąp brakujące wartości w kolumnach liczbowych
    numeric_cols = ['squareMeters', 'rooms', 'buildYear']
    df[numeric_cols] = SimpleImputer(strategy='mean').fit_transform(df[numeric_cols])

    # Rozdzielenie na X (cechy) oraz y (ceny)
    X = df[['squareMeters', 'rooms', 'city', 'buildYear', 'type']].values
    y = df['price'].values

    # Standaryzacja cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

