from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Słownik nazw algorytmów do instancji modeli
ALGORITHMS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=15,
        random_state=42
    ),
    "SVM": LinearSVC(max_iter=1000, dual=False),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
     "Neural Network": MLPRegressor(
         hidden_layer_sizes=(64, 32),  # liczba warstw/neuronów (możesz zmieniać)
         activation='relu',  # relu najlepszy do regresji
         solver='adam',  # adam działa szybciej i lepiej domyślnie
         alpha=1e-3,  # regularization, możesz zostawić lub dać domyślne (1e-4)
         max_iter=400,  # ile maksymalnie epok
         early_stopping=True,  # zatrzymanie, gdy brak postępu
         n_iter_no_change=15,  # ile epok bez poprawy
         random_state=42,
         verbose=True
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=10,
        random_state=42
    )


}

