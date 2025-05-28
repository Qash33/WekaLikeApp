from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Słownik nazw algorytmów do instancji modeli
ALGORITHMS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=10,
        random_state=42
    ),
    "SVM": LinearSVC(max_iter=1000, dual=False),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
     "Neural Network": MLPClassifier(
        hidden_layer_sizes=(32,),  # Prosta sieć z jedną warstwą i 32 neuronami
        max_iter=50,  # 50 iteracji jest wystarczające przy dużych danych
        batch_size=256,  # Większy batch size
        solver='adam',
        learning_rate_init=0.005,  # Stabilna szybkość uczenia
        early_stopping=True,  # Zatrzymanie po braku poprawy
        validation_fraction=0.1,  # 10% danych walidacyjnych
        n_iter_no_change=5,  # Przerwij po 5 epokach bez poprawy
        random_state=42  # Ustalona losowość
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=10,
        random_state=42
    )


}

