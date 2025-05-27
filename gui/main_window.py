from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QPushButton
from PyQt5.QtGui import QFont
from gui.stats_tab import StatsPlotTab
from gui.widgets import create_button, create_combo_box
from model.training import TrainModelThread
from model.preprocess import preprocess_data
from model.algorithms import ALGORITHMS
from model.model_manager import save_model
from utils1.plot_utils import plot_results
from PyQt5.QtWidgets import QFileDialog, QLabel, QProgressBar, QTextEdit, QScrollArea, QHBoxLayout, QMessageBox, QLabel, QLineEdit, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pandas as pd
import numpy as np
import os
import joblib

class WekaLikeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.model = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ML Visualisation & Statistics App")
        self.setGeometry(100, 100, 900, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        mainLayout = QVBoxLayout()
        self.tabs = QTabWidget()
        mainLayout.addWidget(self.tabs)

        self.ml_tab = QWidget()
        ml_layout = QVBoxLayout()

        self.load_model_button = create_button("Load Pretrained Model", "#9C27B0", self.loadPretrainedModel)
        ml_layout.addWidget(self.load_model_button)

        self.load_button = create_button("Load Dataset", "#4CAF50", self.loadDataset)
        ml_layout.addWidget(self.load_button)

        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_label.setFont(QFont("Arial", 12))
        ml_layout.addWidget(self.algorithm_label)

        self.algorithm_combo = create_combo_box(list(ALGORITHMS.keys()))
        ml_layout.addWidget(self.algorithm_combo)

        self.train_button = create_button("Train Model", "#008CBA", self.trainModel)
        ml_layout.addWidget(self.train_button)

        self.save_model_button = create_button("Save Model", "#03A9F4", self.saveModel)
        ml_layout.addWidget(self.save_model_button)

        self.generate_plot_button = QPushButton("Generate Plot")
        self.generate_plot_button.clicked.connect(self.generate_plot)
        self.generate_plot_button.setEnabled(False)
        ml_layout.addWidget(self.generate_plot_button)

        self.progress_bar = QProgressBar()
        ml_layout.addWidget(self.progress_bar)

        self.result_area = QScrollArea()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_area.setWidget(self.result_text)
        ml_layout.addWidget(self.result_area)

        self.chart_combo = create_combo_box(["Scatter Plot", "Histogram", "Line Chart"])
        self.chart_combo.currentIndexChanged.connect(self.handleChartChange)
        ml_layout.addWidget(self.chart_combo)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        ml_layout.addWidget(self.plot_label)

        bottom_layout = QHBoxLayout()
        self.clear_button = create_button("Clear", "#FFA500", self.clearResults)
        self.exit_button = create_button("Exit", "#FF5733", self.close)
        bottom_layout.addWidget(self.clear_button)
        bottom_layout.addWidget(self.exit_button)
        ml_layout.addLayout(bottom_layout)

        self.ml_tab.setLayout(ml_layout)
        self.tabs.addTab(self.ml_tab, "ML Model Visualisation")

        self.stats_tab = StatsPlotTab()
        self.tabs.addTab(self.stats_tab, "Statistical Plots")

        self.central_widget.setLayout(mainLayout)
        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Load Dataset", self.loadDataset)
        file_menu.addAction("Exit", self.close)

        model_menu = menubar.addMenu("Model")
        model_menu.addAction("Train Model", self.trainModel)
        model_menu.addAction("Load Model", self.loadPretrainedModel)
        model_menu.addAction("Save Model", self.saveModel)
        model_menu.addAction("Clear Results", self.clearResults)

        viz_menu = menubar.addMenu("Visualization")
        viz_menu.addAction("Scatter Plot", lambda: self.chart_combo.setCurrentText("Scatter Plot"))
        viz_menu.addAction("Histogram", lambda: self.chart_combo.setCurrentText("Histogram"))
        viz_menu.addAction("Line Chart", lambda: self.chart_combo.setCurrentText("Line Chart"))

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.showAboutDialog)

    def loadPretrainedModel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select model file", "", "All Files (*);;Pickle Files (*.pkl)"
        )
        if file_path:
            try:
                self.model = joblib.load(file_path)
                self.result_text.append(f"Pretrained model loaded from: {file_path}")
                QMessageBox.information(self, "Model loaded", "Pretrained model was loaded successfully!")
                self.check_enable_generate_plot()
            except Exception as e:
                QMessageBox.critical(self, "Loading error", f"Failed to load model: {e}")
                self.result_text.append(f"Error loading model: {e}")

    def add_filter_ui(self):
        filter_layout = QVBoxLayout()

        self.city_filter = QLineEdit()
        self.city_filter.setPlaceholderText("Filter by city...")
        filter_layout.addWidget(QLabel("City Filter"))
        filter_layout.addWidget(self.city_filter)

        self.min_price_filter = QLineEdit()
        self.min_price_filter.setPlaceholderText("Minimum Price...")
        filter_layout.addWidget(QLabel("Minimum Price"))
        filter_layout.addWidget(self.min_price_filter)

        self.max_price_filter = QLineEdit()
        self.max_price_filter.setPlaceholderText("Maximum Price...")
        filter_layout.addWidget(QLabel("Maximum Price"))
        filter_layout.addWidget(self.max_price_filter)

        return filter_layout

    def showAboutDialog(self):
        QMessageBox.about(self, "ML Analyze App", "ML Analyze App\nVersion 1.0\nCreated with PyQt5 and scikit-learn.")

    def loadDataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_path:
            if file_path.endswith(".csv"):
                self.dataset = pd.read_csv(file_path)
            else:
                self.dataset = pd.read_excel(file_path)

            if 'id' in self.dataset.columns:
                self.dataset.drop(columns=['id'], inplace=True)
            self.dataset.insert(0, 'id', range(1, len(self.dataset) + 1))
            self.result_text.append(f"Loaded dataset: {file_path.split('/')[-1]}")
            self.check_enable_generate_plot()

    def handleChartChange(self):
        if self.dataset is None:
            self.result_text.append("Please load a dataset first before changing the chart type.\n")
            self.chart_combo.setCurrentIndex(0)
            return
        chart_type = self.chart_combo.currentText()
        self.result_text.append(f"Chart type changed to: {chart_type}")

    def trainModel(self):
        if self.dataset is None or not isinstance(self.dataset, pd.DataFrame) or self.dataset.empty:
            self.result_text.append("Brak zbioru danych! Proszę załadować dane przed rozpoczęciem treningu.\n")
            return
        try:
            X, y = preprocess_data(self.dataset)
            algorithm_name = self.algorithm_combo.currentText()
            self.model = ALGORITHMS[algorithm_name]
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y, cv=cv, scoring="neg_mean_absolute_error")
            mean_mae = -scores.mean()
            std_mae = scores.std()
            self.result_text.append(
                f"Cross-validation (5-fold):\nMean MAE: {mean_mae:.2f}, Std: {std_mae:.2f}\n"
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            algorithm_name = self.algorithm_combo.currentText()
            self.model = ALGORITHMS[algorithm_name]
            self.train_thread = TrainModelThread(self.model, X_train, X_test, y_train, y_test)
            self.train_thread.progress.connect(self.progress_bar.setValue)
            self.train_thread.result.connect(self.result_text.append)
            self.train_thread.start()
            save_model(self.model, f"data/models/{algorithm_name}.joblib")
            self.check_enable_generate_plot()
        except ValueError as e:
            self.result_text.setPlainText(f"Data processing error: {e}")
        except Exception as e:
            self.result_text.setPlainText(f"Unexpected issue: {e}")

    def saveModel(self):
        if self.model is None:
            QMessageBox.warning(self, "No model", "No model to save! Please train or load a model first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save model as", "", "Pickle Files (*.pkl);;Joblib Files (*.joblib);;All Files (*)"
        )
        if file_path:
            try:
                joblib.dump(self.model, file_path)
                self.result_text.append(f"Model saved to: {file_path}")
                QMessageBox.information(self, "Success", "Model saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save model:\n{e}")
                self.result_text.append(f"Error saving model: {e}")

    def generate_plot(self):
        if (
                self.dataset is None
                or self.model is None
                or not isinstance(self.dataset, pd.DataFrame)
                or self.dataset.empty
        ):
            QMessageBox.warning(self, "Brak danych", "Najpierw załaduj zbiór danych i wytrenuj lub załaduj model!")
            return
        try:
            X, y = preprocess_data(self.dataset)
            if X is None or y is None:
                QMessageBox.warning(self, "Błąd danych", "Błędny format zbioru danych.")
                return
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            predictions = self.model.predict(X_test)
            plot_results(y_test, predictions, self.chart_combo, self.plot_label)
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie można wygenerować wykresu: {e}")

    def clearResults(self):
        self.result_text.clear()
        self.plot_label.clear()
        self.model = None
        self.dataset = None
        self.generate_plot_button.setEnabled(False)

    def check_enable_generate_plot(self):
        # Przycisk aktywny tylko jeśli masz dane i model
        if self.dataset is not None and self.model is not None:
            self.generate_plot_button.setEnabled(True)
        else:
            self.generate_plot_button.setEnabled(False)