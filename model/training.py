from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import mean_absolute_error, r2_score
import time

# Wątek do trenowania modelu w tle, z obsługą sygnałów dla GUI
class TrainModelThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(str)
    plot_signal = pyqtSignal(list, list)

    def __init__(self, model, X_train, X_test, y_train, y_test):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        for i in range(1, 101, 5):
            self.progress.emit(i)
            time.sleep(0.1)  # Symulacja czasu trenowania

        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        self.result.emit(f"Model trained: {type(self.model).__name__}\nMAE: {mae:.2f}\nR²: {r2:.2f}")
        self.plot_signal.emit(self.y_test.tolist(), predictions.tolist())
        self.progress.emit(100)

