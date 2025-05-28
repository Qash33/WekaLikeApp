import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, encoders=None, fit_encoders=True):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    required_columns = [
        'price', 'squareMeters', 'rooms', 'city', 'buildYear', 'type',
        'centreDistance', 'floor', 'floorCount', 'condition', 'buildingMaterial', 'ownership',
        'latitude', 'longitude', 'poiCount', 'hasParkingSpace' ,'hasBalcony' ,'hasElevator' ,'hasSecurity' ,'hasStorageRoom'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolumna {col} jest wymagana w zbiorze danych!")

    # Kolumny kategoryczne
    category_cols = ['city', 'type', 'condition', 'buildingMaterial', 'ownership', 'hasParkingSpace' ,'hasBalcony' ,'hasElevator' ,'hasSecurity' ,'hasStorageRoom']
    # Zamień nietypowe wartości i NaN na 'Brak'
    for col in category_cols:
        df[col] = (
            df[col]
            .replace(['nan', 'NaN', 'n/a', 'N/A', np.nan, None, ''], 'Brak')
            .astype(str)
        )
    # Zakoduj wszystkie kategorie liczbami
    encoders = encoders or {}
    for col in category_cols:
        if fit_encoders or col not in encoders:
            le = LabelEncoder()
            df[col] = df[col].replace([...], 'Brak').astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = df[col].replace([...], 'Brak').astype(str)
            df[col] = encoders[col].transform(df[col])

    # Kolumny liczbowe
    numeric_cols = [
        'squareMeters', 'rooms', 'buildYear', 'centreDistance', 'floor', 'floorCount', 'latitude', 'longitude', 'poiCount'
    ]
    df[numeric_cols] = SimpleImputer(strategy='mean').fit_transform(df[numeric_cols])

    # Ustaw kolumny cech
    feature_cols = [
        'squareMeters', 'rooms', 'city', 'buildYear', 'type',
        'centreDistance', 'floor', 'floorCount', 'condition', 'buildingMaterial', 'ownership',
        'latitude', 'longitude', 'poiCount', 'hasParkingSpace' ,'hasBalcony' ,'hasElevator' ,'hasSecurity' ,'hasStorageRoom'
    ]
    # Upewnij się, że wszystkie kolumny są float (nawet jeśli to int)
    X = df[feature_cols].astype(float)
    y = df['price'].values

    # Standaryzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(df[['city', 'price']].head(15))
    print(df['price'].isna().sum(), "NaN w price po filtrze")

    return X_scaled, y, feature_cols, encoders