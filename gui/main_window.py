from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QPushButton, QFileDialog,
    QLabel, QProgressBar, QTextEdit, QScrollArea, QHBoxLayout, QMessageBox,
    QComboBox
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from gui.stats_tab import StatsPlotTab
from gui.widgets import create_button, create_combo_box
from model.training import TrainModelThread
from model.preprocess import preprocess_data
from model.algorithms import ALGORITHMS
from model.model_manager import save_model
from utils1.plot_utils import plot_results
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pandas as pd
import joblib
import os

class WekaLikeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.model = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ML Visualisation & Statistics App")
        self.setGeometry(100, 100, 1100, 800)

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

        # --- Nowe: selektory filtrów ---
        ml_layout.addWidget(QLabel("Wybierz miasto:"))
        self.city_combo = QComboBox()
        self.city_combo.setEnabled(False)
        ml_layout.addWidget(self.city_combo)

        ml_layout.addWidget(QLabel("Wybierz rok:"))
        self.year_combo = QComboBox()
        self.year_combo.setEnabled(False)
        ml_layout.addWidget(self.year_combo)

        ml_layout.addWidget(QLabel("Wybierz miesiąc:"))
        self.month_combo = QComboBox()
        self.month_combo.setEnabled(False)
        ml_layout.addWidget(self.month_combo)

        ml_layout.addWidget(QLabel("Wybierz parametr liczbowy do analizy:"))
        self.param_combo = QComboBox()
        self.param_combo.setEnabled(False)
        ml_layout.addWidget(self.param_combo)
        # --- Koniec nowych selektorów ---

        self.generate_plot_button = QPushButton("Generuj wykres i dane")
        self.generate_plot_button.clicked.connect(self.generate_filtered_plot)
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
        self.chart_combo.currentIndexChanged.connect(self.generate_filtered_plot)
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
            self.update_filter_selectors()
            self.check_enable_generate_plot()

    def update_filter_selectors(self):
        # Miasto
        self.city_combo.clear()
        self.city_combo.setEnabled(True)
        self.city_combo.addItem("Wszystkie")
        if "city" in self.dataset.columns:
            for val in sorted(self.dataset["city"].dropna().unique()):
                self.city_combo.addItem(str(val))

        # Rok
        self.year_combo.clear()
        self.year_combo.setEnabled(True)
        self.year_combo.addItem("Wszystkie")
        if "year" in self.dataset.columns:
            for val in sorted(self.dataset["year"].dropna().unique()):
                self.year_combo.addItem(str(val))

        # Miesiąc
        self.month_combo.clear()
        self.month_combo.setEnabled(True)
        self.month_combo.addItem("Wszystkie")
        if "month" in self.dataset.columns:
            for val in sorted(self.dataset["month"].dropna().unique()):
                self.month_combo.addItem(str(val))

        # Parametry (liczbowe) wszystkie numeryczne poza id, price, year, month, city
        self.param_combo.clear()
        blacklist = ["id", "city", "year", "month", "price"]
        self.param_combo.setEnabled(True)
        numeric_cols = self.dataset.select_dtypes(include='number').columns
        for col in numeric_cols:
            if col not in blacklist:
                self.param_combo.addItem(col)

    def trainModel(self):
        if self.dataset is None or not isinstance(self.dataset, pd.DataFrame) or self.dataset.empty:
            self.result_text.append("Brak zbioru danych! Proszę załadować dane przed rozpoczęciem treningu.\n")
            return
        try:
            X, y, feature_cols, encoders = preprocess_data(self.dataset)
            self.model_features = feature_cols # <-- ZAPISUJESZ LISTĘ CECH!
            self.model_encoders = encoders
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

    def generate_filtered_plot(self):
        if self.dataset is None or self.dataset.empty:
            QMessageBox.warning(self, "Brak danych", "Najpierw załaduj zbiór danych!")
            return
        if self.model is None:
            QMessageBox.warning(self, "Brak modelu", "Najpierw wytrenuj lub załaduj model!")
            return

        city = self.city_combo.currentText()
        year = self.year_combo.currentText()
        month = self.month_combo.currentText()
        chart_type = self.chart_combo.currentText()

        # Filtrowanie po oryginalnych nazwach (stringach)
        df = self.dataset.copy()
        if city != "Wszystkie" and city != "":
            df = df[df["city"] == city]
        if year != "Wszystkie" and year != "":
            if "year" in df.columns:
                df = df[df["year"].astype(str) == year]
        if month != "Wszystkie" and month != "":
            if "month" in df.columns:
                df = df[df["month"].astype(str) == month]

        # Usuń rekordy bez ceny
        df = df[df['price'].notna()]
        if df.empty:
            QMessageBox.warning(self, "Brak danych", "Brak rekordów z niepustą ceną po filtracji!")
            return

        # Preprocessing – użyj tych samych encoderów co przy trenowaniu (fit_encoders = False)
        encoders = getattr(self, 'model_encoders', None)
        feature_cols = getattr(self, 'model_features', None)
        if encoders is None or feature_cols is None:
            QMessageBox.warning(self, "Brak cech/encoderów", "Brak listy cech lub encoderów z trenowania!")
            return

        from model.preprocess import preprocess_data
        X_scaled, y, feature_cols, _ = preprocess_data(
            df,
            encoders=encoders,
            fit_encoders=False
        )

        # Predykcja modelu
        try:
            y_pred = self.model.predict(X_scaled)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))

            if chart_type == "Scatter Plot":
                plt.scatter(y, y_pred, alpha=0.7)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Idealna predykcja')
                plt.xlabel('Cena rzeczywista')
                plt.ylabel('Cena przewidywana')
                plt.title(f'Predykcja ceny - Scatter Plot\n({city}, {year}, {month})')
                plt.legend()
            elif chart_type == "Histogram":
                plt.hist(y, bins=20, alpha=0.5, label='Cena rzeczywista')
                plt.hist(y_pred, bins=20, alpha=0.5, label='Predykcja')
                plt.xlabel('Cena')
                plt.ylabel('Liczba')
                plt.title(f'Predykcja ceny - Histogram\n({city}, {year}, {month})')
                plt.legend()
            elif chart_type == "Line Chart":
                plt.plot(range(len(y)), y, label='Cena rzeczywista', marker='o')
                plt.plot(range(len(y)), y_pred, label='Predykcja', marker='x')
                plt.xlabel('Rekord')
                plt.ylabel('Cena')
                plt.title(f'Predykcja ceny - Line Chart\n({city}, {year}, {month})')
                plt.legend()
            else:
                plt.scatter(y, y_pred, alpha=0.7)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Idealna predykcja')
                plt.xlabel('Cena rzeczywista')
                plt.ylabel('Cena przewidywana')
                plt.title(f'Predykcja ceny - Scatter Plot\n({city}, {year}, {month})')
                plt.legend()

            plt.tight_layout()
            import os
            output_dir = "data/plots"
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "filtered_pred_vs_true.png")
            plt.savefig(plot_path)
            plt.close()

            from PyQt5.QtGui import QPixmap
            self.plot_label.setPixmap(QPixmap(plot_path))
            self.plot_label.setFixedSize(900, 500)
            self.plot_label.setScaledContents(True)

            # Statystyki tekstowe
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            summary = (
                f"Statystyki ceny (po filtracji):\n"
                f"Liczba rekordów: {len(df)}\n"
                f"MAE: {mae:.2f}\n"
                f"RMSE: {rmse:.2f}\n"
                f"R²: {r2:.2f}\n\n"
                f"Opis cen rzeczywistych:\n{pd.Series(y).describe().to_string()}"
            )
            self.result_text.setPlainText(summary)

        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie można wygenerować wykresu: {e}")

    def clearResults(self):
        self.result_text.clear()
        self.plot_label.clear()
        self.model = None
        self.dataset = None
        self.city_combo.clear()
        self.year_combo.clear()
        self.month_combo.clear()
        self.param_combo.clear()
        self.generate_plot_button.setEnabled(False)

    def check_enable_generate_plot(self):
        if self.dataset is not None and not self.dataset.empty:
            self.generate_plot_button.setEnabled(True)
            self.city_combo.setEnabled(True)
            self.year_combo.setEnabled(True)
            self.month_combo.setEnabled(True)
            self.param_combo.setEnabled(True)
        else:
            self.generate_plot_button.setEnabled(False)