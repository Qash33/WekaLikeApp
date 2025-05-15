import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget,
QComboBox, QTextEdit, QScrollArea, QSizePolicy, QHBoxLayout, QProgressBar, QSplitter, QMessageBox, QTabWidget)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import time
from lightgbm import LGBMClassifier


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
            time.sleep(0.1)

        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)

        self.result.emit(f"Model trained: {type(self.model).__name__}\nAccuracy: {accuracy:.2f}\n")
        self.plot_signal.emit(self.y_test.tolist(), predictions.tolist())
        self.progress.emit(100)

class StatsPlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df_merged = pd.DataFrame()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(3)

        self.load_button = self.createButton("Load Apartment Data", "#4CAF50", self.loadApartmentData)
        layout.addWidget(self.load_button)

        self.info_label = QLabel("Basic Data Info:")
        self.info_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.info_label)

        self.info_text = QTextEdit()
        self.info_text.setFont(QFont("Courier", 10))
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(150)
        layout.addWidget(self.info_text)

        self.missing_label = QLabel("Missing Values:")
        self.missing_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.missing_label)

        self.missing_text = QTextEdit()
        self.missing_text.setFont(QFont("Courier", 10))
        self.missing_text.setReadOnly(True)
        self.missing_text.setFixedHeight(100)
        layout.addWidget(self.missing_text)

        self.stats_label = QLabel("Descriptive Statistics:")
        self.stats_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.stats_label)

        self.stats_text = QTextEdit()
        self.stats_text.setFont(QFont("Courier", 10))
        self.stats_text.setReadOnly(True)
        self.stats_text.setFixedHeight(200)
        layout.addWidget(self.stats_text)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)

        self.cat_label = QLabel("Categorical variable (X axis):")
        self.cat_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.cat_label)

        self.cat_combo = QComboBox()
        self.cat_combo.setFont(QFont("Arial", 11))
        self.cat_combo.currentIndexChanged.connect(self.checkDatasetLoaded)
        layout.addWidget(self.cat_combo)

        self.num_label = QLabel("Numeric variable (Y axis):")
        self.num_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.num_label)

        self.num_combo = QComboBox()
        self.num_combo.setFont(QFont("Arial", 11))
        self.num_combo.currentIndexChanged.connect(self.checkDatasetLoaded)
        layout.addWidget(self.num_combo)

        self.plot_type_label = QLabel("Select plot type:")
        self.plot_type_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.plot_type_label)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.setFont(QFont("Arial", 11))
        self.plot_type_combo.addItems(["Boxplot", "Scatter Plot", "Barplot", "Histogram", "Line Chart"])
        layout.addWidget(self.plot_type_combo)
        layout.addSpacing(10)

        self.generate_initial_plots_button = self.createButton("Generate Initial Plots", "#FFA500", self.generateInitialPlots)
        layout.addWidget(self.generate_initial_plots_button)
        layout.addSpacing(15)

        self.plot_button = self.createButton("Generate Custom Plot", "#008CBA", self.generateStatsPlot)
        layout.addWidget(self.plot_button)
        layout.addSpacing(15)

        exit_layout = QHBoxLayout()
        exit_layout.addStretch()
        self.exit_button = self.createButton("Exit", "#FF5733", QApplication.quit)
        self.exit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        exit_layout.addWidget(self.exit_button)
        exit_layout.addStretch()
        layout.addLayout(exit_layout)

        self.setLayout(layout)

        self.cat_combo.setEnabled(False)
        self.num_combo.setEnabled(False)
        self.generate_initial_plots_button.setEnabled(False)

    def createButton(self, text, color, function):
        button = QPushButton(text)
        button.setFont(QFont("Arial", 12))
        button.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; border-radius: 5px;")
        button.clicked.connect(function)
        return button

    def loadApartmentData(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory Containing CSV Files")
        if folder_path:
            try:
                files = sorted(glob.glob(f"{folder_path}/apartments_pl_20*.csv"))
                files_2023 = [f for f in files if "apartments_pl_2023_" in f]
                files_2024 = [f for f in files if "apartments_pl_2024_" in f]

                dfs_2023 = [pd.read_csv(file) for file in files_2023]
                dfs_2024 = [pd.read_csv(file) for file in files_2024]

                self.df_merged = pd.concat(dfs_2023 + dfs_2024, ignore_index=True)

                self.info_text.setText(self.df_merged.info(buf=pd.io.common.StringIO()).getvalue())

                missing_values = self.df_merged.isnull().sum()
                missing_str = "\n".join(f"{col}: {count}" for col, count in missing_values[missing_values > 0].items())
                self.missing_text.setText(missing_str if missing_str else "No missing values.")

                self.stats_text.setText(self.df_merged.describe().to_string())

                numeric_cols = self.df_merged.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = self.df_merged.select_dtypes(include=['object', 'category']).columns.tolist()

                self.cat_combo.clear()
                self.num_combo.clear()

                if categorical_cols:
                    self.cat_combo.addItems(categorical_cols)
                    self.cat_combo.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Warning", "No suitable categorical columns found.")

                if numeric_cols:
                    self.num_combo.addItems(numeric_cols)
                    self.num_combo.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Warning", "No suitable numeric columns found.")

                self.generate_initial_plots_button.setEnabled(True)
                QMessageBox.information(self, "Success", "Apartment data loaded successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load apartment data: {e}")
                self.df_merged = pd.DataFrame()
                self.cat_combo.clear()
                self.num_combo.clear()
                self.cat_combo.setEnabled(False)
                self.num_combo.setEnabled(False)
                self.generate_initial_plots_button.setEnabled(False)

    def generateInitialPlots(self):
        if self.df_merged.empty:
            QMessageBox.warning(self, "Warning", "Please load apartment data first!")
            return

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df_merged['price'], bins=30, kde=True)
        plt.xlabel("Cena")
        plt.ylabel("Liczba nieruchomości")
        plt.title("Rozkład cen nieruchomości")
        plt.savefig("price_distribution.png", bbox_inches='tight')
        plt.close()
        pixmap = QPixmap("price_distribution.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)
        QMessageBox.information(self, "Success", "Initial price distribution plot generated.")

    def generateStatsPlot(self):
        if self.df_merged.empty:
            QMessageBox.warning(self, "Warning", "Please load apartment data first!")
            return

        cat_var = self.cat_combo.currentText()
        num_var = self.num_combo.currentText()

        if not cat_var or not num_var:
            QMessageBox.warning(self, "Warning", "Please select both a categorical and a numeric variable!")
            return

        plot_type = self.plot_type_combo.currentText()

        plt.figure(figsize=(10, 6))

        try:
            if plot_type == "Boxplot":
                sns.boxplot(data=self.df_merged, x=cat_var, y=num_var, palette="Blues_d")
            elif plot_type == "Scatter Plot":
                sns.scatterplot(data=self.df_merged, x=cat_var, y=num_var, color="purple")
            elif plot_type == "Barplot":
                sns.barplot(data=self.df_merged, x=cat_var, y=num_var, palette="pastel", errorbar=None)
            elif plot_type == "Histogram":
                sns.histplot(data=self.df_merged, x=self.df_merged[num_var], hue=cat_var, multiple="stack", palette="bright", shrink=0.8)
            elif plot_type == "Line Chart":
                sns.lineplot(data=self.df_merged, x=cat_var, y=num_var, marker="o", color="blue")

            plt.title(f"{plot_type} of {num_var} by {cat_var}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("custom_stats_plot.png", bbox_inches='tight')
            plt.close()

            pixmap = QPixmap("custom_stats_plot.png")
            self.plot_label.setPixmap(pixmap)
            self.plot_label.setScaledContents(True)

        except KeyError:
            QMessageBox.critical(self, "Error", "Selected column not found in the dataset.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating plot: {e}")

    def checkDatasetLoaded(self):
        if self.df_merged.empty:
            self.cat_combo.blockSignals(True)
            self.num_combo.blockSignals(True)

            QMessageBox.warning(self, "Warning", "Please load apartment data before selecting variables!")

            self.cat_combo.setCurrentIndex(-1)
            self.num_combo.setCurrentIndex(-1)

            self.cat_combo.setEnabled(False)
            self.num_combo.setEnabled(False)

            self.cat_combo.blockSignals(False)
            self.num_combo.blockSignals(False)


class WekaLikeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.dataset_name = None
        self.model = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ML Visualisation & Statistics App")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            QScrollArea {
                background-color: transparent;
                border: none;
            }

            QTextEdit {
                background-color: #ffffff;
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #cceeff;
                selection-color: #333;
            }

            QComboBox {
                border: 2px solid #aaa;
                background-color: #ffffff;
                border-radius: 8px;
                padding: 5px;
                padding-left: 8px;
                color: #444;
            }

            QComboBox:hover {
                border-color: #555;
            }

            QComboBox:drop-down {
                border-left: 2px solid #aaa;
                width: 30px;
            }

            QComboBox:down-arrow {
                width: 10px;
                height: 10px;
                image: url(://qt-project.org/styles/commonstyle/images/down-32.png);
            }

            QComboBox QAbstractItemView {
                border: 2px solid #aaa;
                background-color: #ffffff;
                selection-background-color: #008CBA;
                selection-color: white;
            }

            QScrollBar:vertical {
                border: none;
                background-color: #f0f0f0;
                width: 12px;
                margin: 15px 3px 15px 3px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical {
                background-color: #888;
                min-height: 30px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #555;
            }

            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                background: none;
            }

            QScrollBar:horizontal {
                border: none;
                background-color: #f0f0f0;
                height: 12px;
                margin: 3px 15px 3px 15px;
                border-radius: 4px;
            }

            QScrollBar::handle:horizontal {
                background-color: #888;
                min-width: 30px;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal:hover {
                background-color: #555;
            }

            QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {
                background: none;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        mainLayout = QVBoxLayout()

        self.tabs = QTabWidget()

        mainLayout.addWidget(self.tabs)

        self.ml_tab = QWidget()
        ml_layout = QVBoxLayout()

        self.load_button = self.createButton("Load Dataset for ML", "#4CAF50", self.loadDataset)
        ml_layout.addWidget(self.load_button)

        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_label.setFont(QFont("Arial", 12, QFont.Bold))
        ml_layout.addWidget(self.algorithm_label)

        self.algorithm_combo = self.createComboBox([
            "Random Forest", "SVM", "Decision Tree", "Neural Network", "LightGBM"
        ])
        ml_layout.addWidget(self.algorithm_combo)

        self.train_button = self.createButton("Train Model", "#008CBA", self.trainModel)
        ml_layout.addWidget(self.train_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        ml_layout.addWidget(self.progress_bar)

        self.result_area = QScrollArea()
        self.result_area.setFixedSize(800, 250)
        self.result_text = QTextEdit()
        self.result_text.setFont(QFont("Courier", 10))
        self.result_text.setReadOnly(True)
        self.result_text.setFixedSize(800, 250)
        self.result_area.setWidget(self.result_text)
        ml_layout.addWidget(self.result_area)

        self.chart_combo = self.createComboBox(["Scatter Plot", "Histogram", "Line Chart"])
        self.chart_combo.currentIndexChanged.connect(self.handleChartChange)
        ml_layout.addWidget(self.chart_combo)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ml_layout.addWidget(self.plot_label)

        bottom_button_layout = QHBoxLayout()

        self.clear_button = self.createButton("Clear", "#FFA500", self.clearResults)
        self.clear_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.exit_button_ml = self.createButton("Exit", "#FF5733", self.close)
        self.exit_button_ml.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bottom_button_layout.addWidget(self.clear_button)
        bottom_button_layout.addWidget(self.exit_button_ml)

        ml_layout.addLayout(bottom_button_layout)

        self.ml_tab.setLayout(ml_layout)
        self.tabs.addTab(self.ml_tab, "ML Model Visualisation")

        self.stats_tab = StatsPlotTab()
        self.tabs.addTab(self.stats_tab, "Statistical Plots")

        self.central_widget.setLayout(mainLayout)

        self.create_menu()

    def handleChartChange(self):
        if self.dataset is None:
            if self.result_text.toPlainText().count("Please load a dataset first before changing the chart type.") == 0:
                self.result_text.append("Please load a dataset first before changing the chart type.\n")
            self.chart_combo.setCurrentIndex(0)
            return

        chart_type = self.chart_combo.currentText()
        self.result_text.append(f"Chart type changed to: {chart_type}")

    def createButton(self, text, color, function):
        button = QPushButton(text)
        button.setFont(QFont("Arial", 12))
        button.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; border-radius: 5px;")
        button.clicked.connect(function)
        return button

    def createComboBox(self, items):
        combo = QComboBox()
        combo.setFont(QFont("Arial", 12))
        combo.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 5px; border-radius: 5px;")
        combo.addItems(items)
        return combo

    def loadDataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File for ML", "",
                                                   "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_path:
            try:
                self.result_text.append(f"Attempting to load dataset for ML from: {file_path}")
                if file_path.endswith(".csv"):
                    self.dataset = pd.read_csv(file_path)
                else:
                    self.dataset = pd.read_excel(file_path)

                if 'id' in self.dataset.columns:
                    self.dataset.drop(columns=['id'], inplace=True)

                self.dataset.insert(0, 'id', range(1, len(self.dataset) + 1))
                self.dataset_name = file_path.split("/")[-1]
                self.result_text.append(f"First few rows:\n{self.dataset.head()}")

            except Exception as e:
                self.result_text.append(f"Error loading dataset for ML: {e}")

    def preprocessData(self):
        if self.dataset is None:
            return None, None

        df = self.dataset.drop(columns=['id'], errors='ignore').copy()
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
        df[cat_cols] = df[cat_cols].apply(lambda col: LabelEncoder().fit_transform(col))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.iloc[:, :-1].values)

        return X_scaled, df.iloc[:, -1].values

    def trainModel(self):
        X, y = self.preprocessData()
        if X is None or y is None:
            self.result_text.append("No dataset loaded or invalid dataset!\n")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        algorithms = {
            "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
            "SVM": LinearSVC(max_iter=1000, verbose=0, dual=False),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=100),
            "LightGBM": LGBMClassifier(n_estimators=50)
        }

        self.model = algorithms[self.algorithm_combo.currentText()]
        self.train_thread = TrainModelThread(self.model, X_train, X_test, y_train, y_test)
        self.train_thread.progress.connect(self.progress_bar.setValue)
        self.train_thread.result.connect(self.result_text.append)
        self.train_thread.plot_signal.connect(self.plotResults)
        self.train_thread.start()

    def plotResults(self, y_test, predictions):
        sample_size = min(1000, len(y_test))
        indices = np.random.choice(range(len(y_test)), sample_size, replace=False) if len(
            y_test) > sample_size else np.arange(len(y_test))

        y_test_sampled = [y_test[i] for i in indices]
        predictions_sampled = [predictions[i] for i in indices]

        plt.figure(figsize=(7.2, 4.8), dpi=100)

        chart_type = self.chart_combo.currentText()

        if chart_type == "Scatter Plot":
            plt.scatter(range(len(y_test_sampled)), y_test_sampled, label='Actual', color='blue', alpha=0.6)
            plt.scatter(range(len(predictions_sampled)), predictions_sampled, label='Predicted', color='red', alpha=0.6)
        elif chart_type == "Histogram":
            plt.hist(y_test_sampled, bins=20, alpha=0.5, label='Actual', color='blue')
            plt.hist(predictions_sampled, bins=20, alpha=0.5, label='Predicted', color='red')
        elif chart_type == "Line Chart":
            plt.plot(y_test_sampled, label='Actual', color='blue')
            plt.plot(predictions_sampled, label='Predicted', color='red', alpha=0.7)

        plt.legend()
        plt.title(f'Actual vs Predicted ({chart_type})')

        plt.savefig("plot.png", bbox_inches='tight', dpi=100)
        plt.close()

        pixmap = QPixmap("plot.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setFixedSize(720, 480)
        self.plot_label.setScaledContents(True)

    def clearResults(self):
        self.result_text.clear()
        self.plot_label.clear()

        if self.dataset is not None:
            self.result_text.append(f"Dataset '{self.dataset_name}' is loaded and ready for training.\n")

    def create_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        load_ml_action = file_menu.addAction("Load Dataset for ML")
        load_ml_action.triggered.connect(self.loadDataset)

        load_stats_action = file_menu.addAction("Load Apartment Data for Stats")
        for i in range(self.tabs.count()):
            if isinstance(self.tabs.widget(i), StatsPlotTab):
                load_stats_action.triggered.connect(self.tabs.widget(i).loadApartmentData)
                break

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        model_menu = menubar.addMenu("Model")

        train_action = model_menu.addAction("Train Model")
        train_action.triggered.connect(self.trainModel)

        clear_results_action = model_menu.addAction("Clear Results")
        clear_results_action.triggered.connect(self.clearResults)

        viz_menu = menubar.addMenu("Visualization")

        scatter_action = viz_menu.addAction("Scatter Plot")
        scatter_action.triggered.connect(lambda: self.chart_combo.setCurrentText("Scatter Plot"))

        histogram_action = viz_menu.addAction("Histogram")
        histogram_action.triggered.connect(lambda: self.chart_combo.setCurrentText("Histogram"))

        line_chart_action = viz_menu.addAction("Line Chart")
        line_chart_action.triggered.connect(lambda: self.chart_combo.setCurrentText("Line Chart"))

        help_menu = menubar.addMenu("Help")

        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.showAboutDialog)

    def showAboutDialog(self):
        QMessageBox.about(self, "ML Analyze App",
                          "ML Analyze App\n"
                          "Version 1.0\n"
                          "Author: Maciej Kuziela\n"
                          "Created with PyQt5 and scikit-learn.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = WekaLikeApp()
    ex.show()
    sys.exit(app.exec_())