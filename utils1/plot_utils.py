# Rysuje wykres wyników modelu i wyświetla go w QLabel
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from PyQt5.QtGui import QPixmap

mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_locale'] = False


# Rysuje wykres wyników modelu i wyświetla go w QLabel
def plot_results(y_test, predictions, chart_combo, plot_label):
    chart_type = chart_combo.currentText()

    if len(y_test) != len(predictions):
        raise ValueError("Length of y_test and predictions must be equal.")

    # Próbkowanie danych (do 1000 próbek, aby uniknąć przeciążenia wykresu)
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(range(len(y_test)), sample_size, replace=False) if len(
        y_test) > sample_size else np.arange(len(y_test))

    y_sample = [y_test[i] for i in indices]
    p_sample = [predictions[i] for i in indices]

    plt.figure(figsize=(10, 6))  # Większy rozmiar wykresu dla czytelności

    if chart_type == "Scatter Plot":
        # Wykres punktowy: Porównanie rzeczywistych i przewidywanych cen
        plt.scatter(range(len(y_sample)), y_sample, label="Ceny rzeczywiste (y_test)", alpha=0.6, color="blue")
        plt.scatter(range(len(p_sample)), p_sample, label="Ceny przewidywane (predictions)", alpha=0.6, color="red")
        plt.xlabel("Indeks próbki (obserwacja)", fontsize=12)
        plt.ylabel("Cena nieruchomości", fontsize=12)

        # Wymuszenie standardowego formatu liczbowego na osi Y
        ax = plt.gca()
        ax.ticklabel_format(style="plain", axis="y")  # Wyłączenie notacji naukowej
        ax.get_yaxis().get_major_formatter().set_scientific(False)  # Dodatkowe ustawienie dla większej pewności

    elif chart_type == "Histogram":
        # Histogram: Rozkład rzeczywistych i przewidywanych cen
        plt.hist(
            y_sample, alpha=0.7, bins=20, label="Ceny rzeczywiste (y_test)", color="blue", edgecolor="black"
        )
        plt.hist(
            p_sample, alpha=0.7, bins=20, label="Ceny przewidywane (predictions)", color="red", edgecolor="black"
        )
        plt.xlabel("Zakres cen nieruchomości", fontsize=12)
        plt.ylabel("Liczba wystąpień", fontsize=12)

        # Usunięcie notacji naukowej i wymuszenie standardowego formatu na osi Y
        ax = plt.gca()
        ax.ticklabel_format(style="plain", axis="y")  # Wyłączenie notacji naukowej
        ax.get_yaxis().get_major_formatter().set_scientific(False)  # Dodatkowe zabezpieczenie
        ax.yaxis.get_offset_text().set_visible(False)  # Ukrycie przesunięcia osi (np. 1e3)

    elif chart_type == "Line Chart":
        # Wykres liniowy: Porównanie rzeczywistych i przewidywanych cen
        plt.plot(
            range(len(y_sample)), y_sample, label="Ceny rzeczywiste (y_test)", color="blue", linestyle="-", alpha=0.8
        )
        plt.plot(
            range(len(p_sample)),
            p_sample,
            label="Ceny przewidywane (predictions)",
            color="red",
            linestyle="--",
            alpha=0.8,
        )
        plt.xlabel("Indeks próbki (obserwacja)", fontsize=12)
        plt.ylabel("Cena nieruchomości", fontsize=12)

        # Wymuszenie standardowego formatu liczbowego na osi Y
        ax = plt.gca()
        ax.ticklabel_format(style="plain", axis="y")  # Wyłączenie notacji naukowej
        ax.get_yaxis().get_major_formatter().set_scientific(False)  # Wyłączenie automatycznego formatowania

    # Dodanie tytułu oraz legendy
    plt.title(f"Porównanie cen rzeczywistych i przewidywanych ({chart_type})", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Zapisanie wykresu w katalogu 'data/plots'
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plot.png"))
    plt.close()

    # Wyświetlenie wykresu w QLabel
    pixmap = QPixmap(os.path.join(output_dir, "plot.png"))
    plot_label.setPixmap(pixmap)
    plot_label.setFixedSize(1080, 720)
    plot_label.setScaledContents(True)
