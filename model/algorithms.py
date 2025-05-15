from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

# Słownik nazw algorytmów do instancji modeli
ALGORITHMS = {
    "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
    "SVM": LinearSVC(max_iter=1000, dual=False),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=100),
    "LightGBM": LGBMClassifier(n_estimators=50)
}
