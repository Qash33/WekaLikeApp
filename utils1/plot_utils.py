import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QPixmap

# Rysuje wykres wyników modelu i wyświetla go w QLabel
def plot_results(y_test, predictions, chart_combo, plot_label):
    chart_type = chart_combo.currentText()

    sample_size = min(1000, len(y_test))
    indices = np.random.choice(range(len(y_test)), sample_size, replace=False) if len(y_test) > sample_size else np.arange(len(y_test))

    y_sample = [y_test[i] for i in indices]
    p_sample = [predictions[i] for i in indices]

    plt.figure(figsize=(8, 5))

    if chart_type == "Scatter Plot":
        plt.scatter(range(len(y_sample)), y_sample, label='Actual', alpha=0.6)
        plt.scatter(range(len(p_sample)), p_sample, label='Predicted', alpha=0.6)
    elif chart_type == "Histogram":
        plt.hist(y_sample, alpha=0.5, label='Actual')
        plt.hist(p_sample, alpha=0.5, label='Predicted')
    elif chart_type == "Line Chart":
        plt.plot(y_sample, label='Actual')
        plt.plot(p_sample, label='Predicted')

    plt.title(f'Actual vs Predicted ({chart_type})')
    plt.legend()
    plt.savefig("data/plots/plot.png")
    plt.close()

    pixmap = QPixmap("data/plots/plot.png")
    plot_label.setPixmap(pixmap)
    plot_label.setFixedSize(720, 480)
    plot_label.setScaledContents(True)