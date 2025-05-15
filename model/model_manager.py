import os
from joblib import dump, load

# Zapisuje model ML do pliku
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)

# Ładuje model ML z pliku (jeśli istnieje)
def load_model(path):
    if os.path.exists(path):
        return load(path)
    return None