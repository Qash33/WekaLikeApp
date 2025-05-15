from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class StatsPlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Przycisk ładowania danych
        self.load_button = QPushButton("Load Dataset")
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.clicked.connect(self.loadDataset)
        layout.addWidget(self.load_button)

        # ComboBox dla zmiennych
        self.cat_combo = QComboBox()
        self.cat_combo.setFont(QFont("Arial", 11))
        layout.addWidget(QLabel("Categorical variable (X axis):"))
        layout.addWidget(self.cat_combo)

        self.num_combo = QComboBox()
        self.num_combo.setFont(QFont("Arial", 11))
        layout.addWidget(QLabel("Numeric variable (Y axis):"))
        layout.addWidget(self.num_combo)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Boxplot", "Scatter Plot", "Barplot", "Histogram", "Line Chart"])
        layout.addWidget(QLabel("Select plot type:"))
        layout.addWidget(self.plot_type_combo)

        self.plot_button = QPushButton("Generate Stats Plot")
        self.plot_button.clicked.connect(self.generateStatsPlot)
        layout.addWidget(self.plot_button)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)

        self.setLayout(layout)

    def loadDataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_path:
            if file_path.endswith(".csv"):
                self.dataset = pd.read_csv(file_path)
            else:
                self.dataset = pd.read_excel(file_path)

            numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.dataset.select_dtypes(include=['object']).columns.tolist()

            self.cat_combo.clear()
            self.cat_combo.addItems(categorical_cols)
            self.num_combo.clear()
            self.num_combo.addItems(numeric_cols)

    def generateStatsPlot(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        cat = self.cat_combo.currentText()
        num = self.num_combo.currentText()
        plot_type = self.plot_type_combo.currentText()

        plt.figure(figsize=(10, 6))

        if plot_type == "Boxplot":
            sns.boxplot(data=self.dataset, x=cat, y=num)
        elif plot_type == "Scatter Plot":
            sns.scatterplot(data=self.dataset, x=cat, y=num)
        elif plot_type == "Barplot":
            sns.barplot(data=self.dataset, x=cat, y=num)
        elif plot_type == "Histogram":
            sns.histplot(data=self.dataset, x=num, hue=cat, multiple="stack")
        elif plot_type == "Line Chart":
            sns.lineplot(data=self.dataset, x=cat, y=num)

        plt.title(f"{plot_type} of {num} by {cat}")
        plt.xticks(rotation=45)
        plt.savefig("data/plots/stats_plot.png")
        plt.close()

        pixmap = QPixmap("data/plots/stats_plot.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)
