import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Wczytywanie plików
files = sorted(glob.glob("apartments_pl_20*_*.csv"))
files_2023 = [f for f in files if "apartments_pl_2023_" in f]
files_2024 = [f for f in files if "apartments_pl_2024_" in f]

dfs_2023 = [pd.read_csv(file) for file in files_2023]
dfs_2024 = [pd.read_csv(file) for file in files_2024]

df_2023 = pd.concat(dfs_2023, ignore_index=True) if dfs_2023 else pd.DataFrame()
df_2024 = pd.concat(dfs_2024, ignore_index=True) if dfs_2024 else pd.DataFrame()
df_merged = pd.concat(dfs_2023 + dfs_2024, ignore_index=True)

# Wyświetlenie podstawowych informacji o danych
df_merged.info()

# Sprawdzenie pierwszych kilku wierszy
print(df_merged.head())

# Sprawdzenie brakujących wartości
missing_values = df_merged.isnull().sum()
print("Brakujące wartości w zbiorze:")
print(missing_values[missing_values > 0])

# Podstawowe statystyki opisowe dla wartości liczbowych
descriptive_stats = df_merged.describe()
print("Statystyki opisowe:")
print(descriptive_stats)

# Wizualizacja rozkładu cen nieruchomości
plt.figure(figsize=(10, 5))
sns.histplot(df_merged['price'], bins=30, kde=True)
plt.xlabel("Cena")
plt.ylabel("Liczba nieruchomości")
plt.title("Rozkład cen nieruchomości")
plt.show()

# Analiza korelacji między cechami numerycznymi
numeric_columns = ['squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 'centreDistance',
                   'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance',
                   'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
                   'pharmacyDistance', 'price']

plt.figure(figsize=(12, 8))
sns.heatmap(df_merged[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Macierz korelacji")
plt.show()

# Analiza wpływu liczby pokoi na cenę
plt.figure(figsize=(10, 5))
sns.boxplot(x='rooms', y='price', data=df_merged)
plt.xlabel("Liczba pokoi")
plt.ylabel("Cena")
plt.title("Wpływ liczby pokoi na cenę nieruchomości")
plt.show()

# Analiza wpływu rodzaju własności na cenę
plt.figure(figsize=(10, 5))
sns.boxplot(x='ownership', y='price', data=df_merged)
plt.xlabel("Rodzaj własności")
plt.ylabel("Cena")
plt.title("Wpływ rodzaju własności na cenę nieruchomości")
plt.xticks(rotation=45)
plt.show()

# Analiza wpływu materiału budynku na cenę
plt.figure(figsize=(10, 5))
sns.boxplot(x='buildingMaterial', y='price', data=df_merged)
plt.xlabel("Materiał budynku")
plt.ylabel("Cena")
plt.title("Wpływ materiału budynku na cenę")
plt.xticks(rotation=45)
plt.show()


# 5️⃣ ANALIZA CZASOWA CEN MIESZKAŃ

# Upewniamy się, że kolumny 'month' i 'year' istnieją i nie mają braków
df_time = df_merged.dropna(subset=['price', 'month', 'year'])

# Tworzymy nową kolumnę łączącą rok i miesiąc
df_time['year_month'] = df_time['year'].astype(int).astype(str) + '-' + df_time['month'].astype(int).astype(str).str.zfill(2)

# 📈 Średnia cena mieszkań w czasie (miesiąc/rok)
monthly_avg = df_time.groupby('year_month')['price'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(x='year_month', y='price', data=monthly_avg, marker='o')
plt.xticks(rotation=45)
plt.title("Średnia cena nieruchomości w czasie (miesiąc/rok)")
plt.xlabel("Data (YYYY-MM)")
plt.ylabel("Średnia cena (PLN)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Średnia cena roczna
yearly_avg = df_time.groupby('year')['price'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='year', y='price', data=yearly_avg)
plt.title("Średnia cena nieruchomości na przestrzeni lat")
plt.xlabel("Rok")
plt.ylabel("Średnia cena (PLN)")
plt.grid(axis='y')
plt.show()

# 📌 Liczba ogłoszeń w czasie (opcjonalnie)
monthly_counts = df_time['year_month'].value_counts().sort_index()

plt.figure(figsize=(14, 6))
sns.barplot(x=monthly_counts.index, y=monthly_counts.values)
plt.xticks(rotation=45)
plt.title("Liczba ogłoszeń nieruchomości w czasie (miesiąc/rok)")
plt.xlabel("Data (YYYY-MM)")
plt.ylabel("Liczba ogłoszeń")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 🏙️ Porównanie średnich cen między miastami w czasie
df_time_city = df_time.dropna(subset=['city'])

# Grupowanie: średnia cena w mieście i miesiącu
avg_city_time = df_time_city.groupby(['city', 'year_month'])['price'].mean().reset_index()

# 🏆 Miasto i czas, kiedy było najtaniej
min_price_row = avg_city_time.loc[avg_city_time['price'].idxmin()]
cheapest_city = min_price_row['city']
cheapest_month = min_price_row['year_month']
cheapest_price = min_price_row['price']

# 🔥 Wykres z adnotacją
plt.figure(figsize=(14, 6))
sns.lineplot(data=avg_city_time, x='year_month', y='price', hue='city', marker='o')
plt.xticks(rotation=45)
plt.title("Średnia cena mieszkań w miastach na przestrzeni czasu")
plt.xlabel("Data (YYYY-MM)")
plt.ylabel("Średnia cena (PLN)")

# Dodajemy adnotację do punktu z najniższą ceną
plt.annotate(
    f"Najlepszy moment:\n{cheapest_city.title()}, {cheapest_month}\n{cheapest_price:,.0f} PLN",
    xy=(cheapest_month, cheapest_price),
    xytext=(cheapest_month, cheapest_price + 50000),
    arrowprops=dict(arrowstyle="->", color='black'),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")
)

plt.grid(True)
plt.tight_layout()
plt.show()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#1️⃣ Przygotowanie danych
#Wybieramy cechy numeryczne i kategoryczne
num_features = ['squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 'centreDistance',
                'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance',
                'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
                'pharmacyDistance']
cat_features = ['ownership', 'buildingMaterial', 'condition']

# Usunięcie wierszy z brakującymi wartościami
df_clean = df_merged.dropna(subset=num_features + ['price'])

# Transformacja cech (standaryzacja numerycznych, kodowanie kategorycznych)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X = df_clean[num_features + cat_features]
y = df_clean['price']

# Przekształcenie danych
X_transformed = preprocessor.fit_transform(X)

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 2️⃣ Budowa sieci neuronowej
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Warstwa wyjściowa do regresji
])

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# 3️⃣ Trenowanie modelu
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# 4️⃣ Ewaluacja modelu
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Średni błąd absolutny (MAE) na zbiorze testowym: {test_mae:.2f} PLN")

# Wizualizacja procesu uczenia
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='MAE - trenowanie')
plt.plot(history.history['val_mae'], label='MAE - walidacja')
plt.xlabel("Epoka")
plt.ylabel("Średni błąd absolutny (PLN)")
plt.title("Proces uczenia sieci neuronowej")
plt.legend()
plt.show()