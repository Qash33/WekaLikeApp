from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

# Słownik nazw algorytmów do instancji modeli
ALGORITHMS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=10,  # Zmniejszono liczbę drzew
        max_depth=10,  # Ograniczenie maksymalnej głębokości drzewa
        n_jobs=1  # Ustawienie na 1, aby uniknąć problemów z wielowątkowością
    ),
    "SVM": LinearSVC(max_iter=1000, dual=False),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
     "Neural Network": MLPClassifier(
        hidden_layer_sizes=(32,),  # Prosta sieć z jedną warstwą i 32 neuronami
        max_iter=50,  # 50 iteracji jest wystarczające przy dużych danych
        batch_size=512,  # Większy batch size
        solver='adam',
        learning_rate_init=0.005,  # Stabilna szybkość uczenia
        early_stopping=True,  # Zatrzymanie po braku poprawy
        validation_fraction=0.1,  # 10% danych walidacyjnych
        n_iter_no_change=5,  # Przerwij po 5 epokach bez poprawy
        random_state=42  # Ustalona losowość
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100,  # Liczba słabych klasyfikatorów
        max_depth=6,  # Maksymalna głębokość drzew
        learning_rate=0.1,  # Szybkość uczenia
        n_jobs=-1,  # Użycie wszystkich dostępnych rdzeni
        random_state=42,  # Powtarzalność wyników
        early_stopping_round=5  # Zamiast 'early_stopping=True' i 'n_iter_no_change=5'
    )


}

