from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib


import matplotlib.pyplot as plt

import os

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
        self.plot_type_combo.addItems(["Boxplot", "Scatter Plot", "Trend Analysis", "Correlation Heatmap", "Histogram"])
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

            # Diagnostyka danych
            print(self.dataset.dtypes)  # Wyświetlanie typów danych w załadowanym zbiorze

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

        if 'id' in self.dataset.columns:
            self.dataset.drop(columns=['id'], inplace=True)

        cat = self.cat_combo.currentText()
        num = self.num_combo.currentText()
        plot_type = self.plot_type_combo.currentText()

        if not cat or not num:
            QMessageBox.warning(self, "Warning", "Please select both categorical and numeric variables.")
            return

        # Czyszczenie danych
        try:
            plot_data = self.dataset[[cat, num]].dropna()
        except KeyError as e:
            QMessageBox.critical(self, "Error", f"Column not found: {e}")
            return

        # Sprawdzenie czy dane są numeryczne
        if not np.issubdtype(plot_data[num].dtype, np.number):
            QMessageBox.critical(self, "Error", f"The column '{num}' must be numeric.")
            return

        # Konwersja kolumny kategorycznej (jeśli trzeba)
        if not pd.api.types.is_categorical_dtype(plot_data[cat]) and not pd.api.types.is_object_dtype(plot_data[cat]):
            plot_data[cat] = plot_data[cat].astype(str)

        output_dir = "data/plots"
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create directory {output_dir}: {e}")
                return

        plt.figure(figsize=(14, 10))

        try:
            if plot_type == "Boxplot":
                sns.boxplot(data=plot_data, x=cat, y=num)
            elif plot_type == "Scatter Plot":
                sns.scatterplot(data=plot_data, x=cat, y=num)
            elif plot_type == "Correlation Heatmap":
                correlation_matrix = self.dataset.corr(numeric_only=True)
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            elif plot_type == "Histogram":
                sns.histplot(data=plot_data, x=num, bins=30)
            elif plot_type == "Trend Analysis":
                if "price" not in self.dataset.columns:
                    QMessageBox.critical(self, "Error", "Column 'price' not found for Trend Analysis.")
                    return
                sns.lineplot(data=self.dataset, x=num, y="price", hue=cat)

            plt.title(f"{plot_type} of {num} by {cat}")
            plt.xticks(rotation=45)

            plot_path = os.path.join(output_dir, "stats_plot.png")
            plt.savefig(plot_path)
            plt.close()

            pixmap = QPixmap(plot_path)
            self.plot_label.setPixmap(pixmap)
            self.plot_label.setScaledContents(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error while generating plot: {e}")



