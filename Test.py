import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget,
                             QComboBox, QTextEdit, QScrollArea, QSizePolicy, QHBoxLayout, QProgressBar, QSplitter,
                             QMessageBox)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import seaborn as sns


class StatsPlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Statistical Plots AI")
        self.setGeometry(100, 100, 900, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()

        self.load_button = self.createButton("Load Dataset", "#4CAF50", self.loadDataset)
        layout.addWidget(self.load_button)

        self.var_combo = QComboBox()
        self.var_combo.setFont(QFont("Arial", 12))
        layout.addWidget(self.var_combo)

        self.plot_button = self.createButton("Generate Stats Plot", "#008CBA", self.generateStatsPlot)
        layout.addWidget(self.plot_button)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)

        self.central_widget.setLayout(layout)
        self.create_menu()

    def createButton(self, text, color, function):
        button = QPushButton(text)
        button.setFont(QFont("Arial", 12))
        button.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; border-radius: 5px;")
        button.clicked.connect(function)
        return button

    def loadDataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_path:
            try:
                if file_path.endswith(".csv"):
                    self.dataset = pd.read_csv(file_path)
                else:
                    self.dataset = pd.read_excel(file_path)

                numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
                self.var_combo.clear()
                self.var_combo.addItems(numeric_cols)
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def generateStatsPlot(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        var = self.var_combo.currentText()
        if var == "":
            QMessageBox.warning(self, "Warning", "Please select a variable!")
            return

        data = self.dataset[var].dropna()
        mean = np.mean(data)
        median = np.median(data)
        mode = data.mode()[0]
        min_val = np.min(data)
        max_val = np.max(data)

        plt.figure(figsize=(7.2, 4.8))
        sns.histplot(data, kde=True, bins=20, color='blue', alpha=0.6)
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean:.2f}")
        plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f"Median: {median:.2f}")
        plt.axvline(mode, color='purple', linestyle='dashed', linewidth=2, label=f"Mode: {mode:.2f}")
        plt.axvline(min_val, color='black', linestyle='dotted', linewidth=2, label=f"Min: {min_val:.2f}")
        plt.axvline(max_val, color='orange', linestyle='dotted', linewidth=2, label=f"Max: {max_val:.2f}")
        plt.legend()
        plt.title(f"Statistical Plot for {var}")

        plt.savefig("stats_plot.png")
        plt.close()

        pixmap = QPixmap("stats_plot.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)

    def create_menu(self):
        menubar = self.menuBar()
        stats_menu = menubar.addMenu("Statistics")
        generate_plot_action = stats_menu.addAction("Generate Stats Plot")
        generate_plot_action.triggered.connect(self.generateStatsPlot)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = StatsPlotApp()
    ex.show()
    sys.exit(app.exec_())
