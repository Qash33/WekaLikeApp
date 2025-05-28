from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QPushButton, QFileDialog,
    QLabel, QProgressBar, QTextEdit, QScrollArea, QHBoxLayout, QMessageBox,
    QComboBox, QSizePolicy
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from qtrangeslider import QRangeSlider
from gui.stats_tab import StatsPlotTab
from gui.widgets import create_button, create_combo_box
from model.training import TrainModelThread
from model.preprocess import preprocess_data
from model.algorithms import ALGORITHMS
from model.model_manager import save_model
from utils1.plot_utils import plot_results
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from scipy.stats import gaussian_kde
import pandas as pd
import joblib
import os

class WekaLikeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.model = None
        self.model_enc = None
        self.model_scaler = None
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

        slider_layout = QHBoxLayout()

        # --- Powierzchnia (metry kw.) ---
        slider_layout.addWidget(QLabel("Metraż:"))
        self.area_slider = QRangeSlider()
        self.area_slider.setOrientation(Qt.Horizontal)
        self.area_slider.setEnabled(False)
        slider_layout.addWidget(self.area_slider)
        self.area_range_label = QLabel("")
        slider_layout.addWidget(self.area_range_label)

        # --- Pokoje ---
        slider_layout.addWidget(QLabel("Pokoje:"))
        self.rooms_slider = QRangeSlider()
        self.rooms_slider.setOrientation(Qt.Horizontal)
        self.rooms_slider.setEnabled(False)
        slider_layout.addWidget(self.rooms_slider)
        self.rooms_range_label = QLabel("")
        slider_layout.addWidget(self.rooms_range_label)

        # --- Rok budowy ---
        slider_layout.addWidget(QLabel("Rok budowy:"))
        self.year_slider = QRangeSlider()
        self.year_slider.setOrientation(Qt.Horizontal)
        self.year_slider.setEnabled(False)
        slider_layout.addWidget(self.year_slider)
        self.year_range_label = QLabel("")
        slider_layout.addWidget(self.year_range_label)

        # Podłącz zmiany wartości suwaków do aktualizacji etykiet
        self.area_slider.valueChanged.connect(self.update_slider_labels)
        self.rooms_slider.valueChanged.connect(self.update_slider_labels)
        self.year_slider.valueChanged.connect(self.update_slider_labels)

        ml_layout.addLayout(slider_layout)

        # --- Koniec nowych selektorów ---

        self.generate_plot_button = QPushButton("Generuj wykres i dane")
        self.generate_plot_button.clicked.connect(self.generate_filtered_plot)
        self.generate_plot_button.setEnabled(False)
        ml_layout.addWidget(self.generate_plot_button)

        self.progress_bar = QProgressBar()
        ml_layout.addWidget(self.progress_bar)

        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_text = QTextEdit()
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_text.setReadOnly(True)
        self.result_area.setWidget(self.result_text)
        ml_layout.addWidget(self.result_area)

        self.chart_combo = create_combo_box(["Scatter Plot", "Fill Between", "Line Chart"])
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
        viz_menu.addAction("Fill Between", lambda: self.chart_combo.setCurrentText("Fill Between"))
        viz_menu.addAction("Line Chart", lambda: self.chart_combo.setCurrentText("Line Chart"))

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.showAboutDialog)

    def loadPretrainedModel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select model file", "", "All Files (*);;Pickle Files (*.pkl);;Joblib Files (*.joblib)"
        )
        if file_path:
            try:
                data = joblib.load(file_path)
                self.model = data["model"]
                self.model_enc = data.get("enc", None)
                self.model_scaler = data.get("scaler", None)
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
            self.set_slider_ranges_from_dataset()
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

    def set_slider_ranges_from_dataset(self):
        df = self.dataset
        # Zakres dla metrażu-mapowanie po indeksie na inty!
        if 'squareMeters' in df.columns:
            self.area_unique = sorted(set(int(round(v)) for v in df['squareMeters'].dropna()))
            self.area_slider.setMinimum(0)
            self.area_slider.setMaximum(len(self.area_unique) - 1)
            self.area_slider.setValue((0, len(self.area_unique) - 1))
            self.area_slider.setEnabled(True)
        # Zakres dla pokoi
        if 'rooms' in df.columns:
            min_val, max_val = int(df['rooms'].min()), int(df['rooms'].max())
            if min_val == max_val:
                min_val = max_val - 1
            self.rooms_slider.setMinimum(min_val)
            self.rooms_slider.setMaximum(max_val)
            self.rooms_slider.setValue((min_val, max_val))
            self.rooms_slider.setEnabled(True)
        # Zakres dla roku budowy-mapowanie po indeksie na inty!
        if 'buildYear' in df.columns:
            self.year_unique = sorted(set(int(v) for v in df['buildYear'].dropna()))
            self.year_slider.setMinimum(0)
            self.year_slider.setMaximum(len(self.year_unique) - 1)
            self.year_slider.setValue((0, len(self.year_unique) - 1))
            self.year_slider.setEnabled(True)
        else:
            self.year_slider.setMinimum(0)
            self.year_slider.setMaximum(99)
            self.year_slider.setValue((0, 99))
            self.year_slider.setEnabled(False)
        self.update_slider_labels()

    def update_slider_labels(self):
        # Metraż
        if hasattr(self, "area_unique") and self.area_unique:
            area_min_idx, area_max_idx = self.area_slider.value()
            a_min = self.area_unique[area_min_idx]
            a_max = self.area_unique[area_max_idx]
            self.area_range_label.setText(f"{a_min} - {a_max} m²")
        else:
            area_min, area_max = self.area_slider.value()
            self.area_range_label.setText(f"{area_min} - {area_max} m²")
        # Pokoje
        rooms_min, rooms_max = self.rooms_slider.value()
        self.rooms_range_label.setText(f"{rooms_min} - {rooms_max} pokoi")
        # Rok budowy
        if hasattr(self, "year_unique") and self.year_unique:
            year_min_idx, year_max_idx = self.year_slider.value()
            y_min = self.year_unique[year_min_idx]
            y_max = self.year_unique[year_max_idx]
            self.year_range_label.setText(f"{y_min} - {y_max} r.")
        else:
            year_min, year_max = self.year_slider.value()
            self.year_range_label.setText(f"{year_min} - {year_max} r.")

    def trainModel(self):
        if self.dataset is None or not isinstance(self.dataset, pd.DataFrame) or self.dataset.empty:
            self.result_text.append("Brak zbioru danych! Proszę załadować dane przed rozpoczęciem treningu.\n")
            return
        try:
            from model.preprocess import preprocess_data
            X, y, feature_cols, enc, scaler = preprocess_data(self.dataset, fit=True)
            self.model_features = feature_cols
            self.model_enc = enc
            self.model_scaler = scaler

            algorithm_name = self.algorithm_combo.currentText()
            self.model = ALGORITHMS[algorithm_name]

            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y, cv=cv, scoring="neg_mean_absolute_error")
            mean_mae = -scores.mean()
            std_mae = scores.std()
            self.result_text.append(
                f"Cross-validation (3-fold):\nMean MAE: {mean_mae:.2f}, Std: {std_mae:.2f}\n"
            )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
                joblib.dump({
                    "model": self.model,
                    "enc": self.model_enc,
                    "scaler": self.model_scaler
                }, file_path)
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

        df = self.dataset.copy()
        if city != "Wszystkie" and city != "":
            df = df[df["city"] == city]
        if year != "Wszystkie" and year != "":
            if "year" in df.columns:
                df = df[df["year"].astype(str) == year]
        if month != "Wszystkie" and month != "":
            if "month" in df.columns:
                df = df[df["month"].astype(str) == month]

        # Filtracja po suwakach
        # Metraż
        if hasattr(self, "area_unique") and self.area_unique:
            area_min_idx, area_max_idx = self.area_slider.value()
            area_min = self.area_unique[area_min_idx]
            area_max = self.area_unique[area_max_idx]
            df = df[(df['squareMeters'].round().astype(int) >= area_min) & (
                        df['squareMeters'].round().astype(int) <= area_max)]
        else:
            area_min, area_max = self.area_slider.value()
            df = df[(df['squareMeters'] >= area_min) & (df['squareMeters'] <= area_max)]
            # Pokoje
        rooms_min, rooms_max = self.rooms_slider.value()
        if 'rooms' in df.columns:
            df = df[(df['rooms'] >= rooms_min) & (df['rooms'] <= rooms_max)]
        # Rok budowy
        if hasattr(self, "year_unique") and self.year_unique and 'buildYear' in df.columns:
            year_min_idx, year_max_idx = self.year_slider.value()
            y_min = self.year_unique[year_min_idx]
            y_max = self.year_unique[year_max_idx]
            df = df[df['buildYear'].notna()]
            df = df[(df['buildYear'].astype(int) >= y_min) & (df['buildYear'].astype(int) <= y_max)]
        # --- Koniec filtracji po suwakach ---

        df = df[df['price'].notna()]
        if df.empty:
            QMessageBox.warning(self, "Brak danych", "Brak rekordów z niepustą ceną po filtracji!")
            return

        # Używamy tych samych encoderów i scalerów co przy treningu
        from model.preprocess import preprocess_data
        X, y_log, feature_cols, _, _ = preprocess_data(
            df,
            enc=self.model_enc,
            scaler=self.model_scaler,
            fit=False
        )

        try:
            y_pred_log = self.model.predict(X)

            # Przekształcenie z log-ceny na normalną cenę
            import numpy as np
            y_real = np.exp(y_log)
            y_pred_real = np.exp(y_pred_log)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))

            if chart_type == "Scatter Plot":
                plt.scatter(y_real, y_pred_real, alpha=0.7)
                plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', label='Idealna predykcja')
                plt.xlabel('Cena rzeczywista [PLN]')
                plt.ylabel('Cena przewidywana [PLN]')
                plt.title(f'Predykcja ceny - Scatter Plot\n({city}, {year}, {month})')
                plt.legend()
            elif chart_type == "Fill Between":

                x = np.linspace(min(y_real.min(), y_pred_real.min()), max(y_real.max(), y_pred_real.max()), 1000)

                # Gęstość KDE
                kde_real = gaussian_kde(y_real)
                kde_pred = gaussian_kde(y_pred_real)
                y_real_kde = kde_real(x)
                y_pred_kde = kde_pred(x)

                plt.figure(figsize=(10, 6))
                plt.fill_between(x, y_real_kde, color="blue", alpha=0.4, label="Cena rzeczywista")
                plt.fill_between(x, y_pred_kde, color="red", alpha=0.4, label="Predykcja")
                plt.plot(x, y_real_kde, color="blue")
                plt.plot(x, y_pred_kde, color="red")
                plt.xlabel('Cena [PLN]')
                plt.ylabel('Gęstość (Ilość wystąpień)')
                plt.title('Porównanie rozkładów cen (fill_between)')
                plt.legend()
                plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
            elif chart_type == "Line Chart":
                plt.plot(range(len(y_real)), y_real, label='Cena rzeczywista', marker='o')
                plt.plot(range(len(y_pred_real)), y_pred_real, label='Predykcja', marker='x')
                plt.xlabel('Rekord')
                plt.ylabel('Cena [PLN]')
                plt.title(f'Predykcja ceny - Line Chart\n({city}, {year}, {month})')
                plt.legend()
            else:
                plt.scatter(y_real, y_pred_real, alpha=0.7)
                plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', label='Idealna predykcja')
                plt.xlabel('Cena rzeczywista [PLN]')
                plt.ylabel('Cena przewidywana [PLN]')
                plt.title(f'Predykcja ceny - Scatter Plot\n({city}, {year}, {month})')
                plt.legend()

            plt.tight_layout()
            output_dir = "data/plots"
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "filtered_pred_vs_true.png")
            plt.savefig(plot_path)
            plt.close()

            self.plot_label.setPixmap(QPixmap(plot_path))
            self.plot_label.setFixedSize(900, 500)
            self.plot_label.setScaledContents(True)

            # Statystyki tekstowe
            y_real = np.exp(y_log)
            y_pred_real = np.exp(y_pred_log)

            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
            mae = mean_absolute_error(y_real, y_pred_real)
            rmse = np.sqrt(mean_squared_error(y_real, y_pred_real))
            r2 = r2_score(y_real, y_pred_real)
            mape = mean_absolute_percentage_error(y_real, y_pred_real) * 100
            desc = pd.Series(y_real).describe()
            desc = desc.apply(lambda x: f"{x:,.0f}").to_string()
            summary = (
                f"Statystyki ceny (po filtracji):\n"
                f"Liczba rekordów: {len(y_real)}\n"
                f"MAE: {mae:,.2f} PLN\n"
                f"RMSE: {rmse:,.2f} PLN\n"
                f"MAPE: {mape:,.2f} %\n"
                f"R²: {r2:.2f}\n\n"
                f"Opis cen rzeczywistych:\n{desc}"
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
        self.generate_plot_button.setEnabled(False)
        self.area_slider.setEnabled(False)
        self.rooms_slider.setEnabled(False)
        self.year_slider.setEnabled(False)
        self.area_range_label.setText("")
        self.rooms_range_label.setText("")
        self.year_range_label.setText("")

    def check_enable_generate_plot(self):
        if self.dataset is not None and not self.dataset.empty:
            self.generate_plot_button.setEnabled(True)
            self.city_combo.setEnabled(True)
            self.year_combo.setEnabled(True)
            self.month_combo.setEnabled(True)
            self.area_slider.setEnabled(True)
            self.rooms_slider.setEnabled(True)
            self.year_slider.setEnabled(True)
        else:
            self.generate_plot_button.setEnabled(False)
            self.area_slider.setEnabled(False)
            self.rooms_slider.setEnabled(False)
            self.year_slider.setEnabled(False)