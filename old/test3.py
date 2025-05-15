import sys
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import time
import numpy as np
from lightgbm import LGBMClassifier
import seaborn as sns


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
        self.dataset = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(3)

        self.load_button = self.createButton("Load Dataset", "#4CAF50", self.loadDataset)
        layout.addWidget(self.load_button)

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
        layout.addSpacing(50)

        self.plot_button = self.createButton("Generate Stats Plot", "#008CBA", self.generateStatsPlot)
        layout.addWidget(self.plot_button)
        layout.addSpacing(15)

        self.summary_plot_button = self.createButton("Generate Summary Plot", "#7DCEA0", self.generateSummaryPlot)
        layout.addWidget(self.summary_plot_button)
        layout.addSpacing(15)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)

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

                if 'id' in self.dataset.columns:
                    self.dataset.drop(columns=['id'], inplace=True)

                self.dataset.insert(0, 'id', range(1, len(self.dataset) + 1))

                numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = self.dataset.select_dtypes(include=['object', 'category']).columns.tolist()

                self.cat_combo.clear()
                self.num_combo.clear()

                if len(categorical_cols) == 0:
                    QMessageBox.warning(self, "Warning", "No suitable categorical columns found for X axis!")
                else:
                    self.cat_combo.addItems(categorical_cols)

                if len(numeric_cols) == 0:
                    QMessageBox.warning(self, "Warning", "No suitable numeric columns found for Y axis!")
                else:
                    self.num_combo.addItems(numeric_cols)

                self.cat_combo.setEnabled(True)
                self.num_combo.setEnabled(True)

                QMessageBox.information(self, "Success", "Dataset loaded successfully! IDs generated automatically.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def generateStatsPlot(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        cat_var = self.cat_combo.currentText()
        num_var = self.num_combo.currentText()

        if cat_var == "" or num_var == "":
            QMessageBox.warning(self, "Warning", "Please select both variables!")
            return

        plot_type = self.plot_type_combo.currentText()

        plt.figure(figsize=(10, 6))

        if plot_type == "Boxplot":
            sns.boxplot(data=self.dataset, x=cat_var, y=num_var, palette="Blues_d")
        elif plot_type == "Scatter Plot":
            sns.scatterplot(data=self.dataset, x=cat_var, y=num_var, color="purple")
        elif plot_type == "Barplot":
            sns.barplot(data=self.dataset, x=cat_var, y=num_var, palette="pastel", errorbar=None)
        elif plot_type == "Histogram":
            sns.histplot(data=self.dataset, x=num_var, hue=cat_var, multiple="stack", palette="bright", shrink=0.8)
        elif plot_type == "Line Chart":
            sns.lineplot(data=self.dataset, x=cat_var, y=num_var, marker="o", color="blue")

        plt.title(f"{plot_type} of {num_var} by {cat_var}")
        plt.xticks(rotation=45)

        plt.savefig("stats_plot.png", bbox_inches='tight')
        plt.close()

        pixmap = QPixmap("stats_plot.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)

    def generateSummaryPlot(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        num_var = self.num_combo.currentText()
        cat_var = self.cat_combo.currentText()

        if num_var == "" or cat_var == "":
            QMessageBox.warning(self, "Warning", "Please select both variables!")
            return

        # Obliczenia statystyczne
        grouped = self.dataset.groupby(cat_var)[num_var]
        stats = grouped.agg(['min', 'max', 'median', 'mean', lambda x: x.mode().iloc[0], 'std'])
        stats.columns = ['Min', 'Max', 'Median', 'Mean', 'Mode', 'StdDev']  # Nazwy kolumn

        stats.reset_index(inplace=True)

        # Budowanie wykresu
        plt.figure(figsize=(10, 6))

        # Wykres słupkowy dla lepszej widoczności statystyk
        sns.barplot(x=cat_var, y='Mean', data=stats, color='lightblue', linewidth=2, edgecolor='black', errorbar=None)

        plt.errorbar(x=range(len(stats)), y=stats['Mean'],
                     yerr=stats['StdDev'], fmt='o', color='black', label='Std. deviation')

        plt.scatter(stats.index, stats['Median'], color='blue', marker='D', s=80, label='Median')
        plt.scatter(stats.index, stats['Mode'], color='red', marker='X', s=80, label='Mode')
        plt.scatter(stats.index, stats['Min'], color='green', marker='v', s=80, label='Minimum')
        plt.scatter(stats.index, stats['Max'], color='purple', marker='^', s=80, label='Maximum')

        plt.title(f"Summary statistics for '{num_var}' by '{cat_var}'")
        plt.xlabel(cat_var)
        plt.ylabel(num_var)
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)

        plt.savefig("summary_stats_plot.png", bbox_inches='tight')
        plt.close()

        pixmap = QPixmap("summary_stats_plot.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)

    def checkDatasetLoaded(self):
        if self.dataset is None:
            self.cat_combo.blockSignals(True)
            self.num_combo.blockSignals(True)

            QMessageBox.warning(self, "Warning", "Please load a dataset before selecting variables!")

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
        self.model = None
        self.initUI()

    def calculate_cross_validation(self, X, y):
        """
        Funkcja do przeprowadzania walidacji krzyżowej dla każdego modelu i porównania wyników.
        """
        self.result_text.append("\n=== Cross-Validation Results ===")
        algorithms = {
            "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
            "SVM": LinearSVC(max_iter=1000, verbose=0, dual=False),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=100),
            "LightGBM": LGBMClassifier(n_estimators=50)
        }

        results = {}
        for name, algorithm in algorithms.items():
            try:
                scores = cross_val_score(algorithm, X, y, cv=5, scoring="accuracy")
                results[name] = scores
                self.result_text.append(
                    f"{name} - Mean Accuracy: {scores.mean():.2f} / Std Dev: {scores.std():.2f}"
                )
            except Exception as e:
                self.result_text.append(f"Error with {name}: {str(e)}")

        # Porównanie wyników w formie tabeli
        self.result_text.append("\nModel Comparison:")
        for name, scores in results.items():
            self.result_text.append(f"{name}: Mean={scores.mean():.2f}, StdDev={scores.std():.2f}")

        # Dodatkowo można zwrócić najlepszy wynik
        best_model = max(results, key=lambda k: results[k].mean())
        self.result_text.append(f"\nBest Model: {best_model} with Accuracy={results[best_model].mean():.2f}")

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

        self.load_button = self.createButton("Load Dataset", "#4CAF50", self.loadDataset)
        ml_layout.addWidget(self.load_button)

        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_label.setFont(QFont("Arial", 12, QFont.Bold))
        ml_layout.addWidget(self.algorithm_label)

        self.algorithm_combo = self.createComboBox([
            "Random Forest", "SVM", "Decision Tree", "Neural Network", "LightGBM"
        ])
        ml_layout.addWidget(self.algorithm_combo)

        # Train Model
        self.train_button = self.createButton("Train Model", "#008CBA", self.trainModel)
        ml_layout.addWidget(self.train_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        ml_layout.addWidget(self.progress_bar)

        # Wyniki tekstowe
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

        self.exit_button = self.createButton("Exit", "#FF5733", self.close)
        self.exit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bottom_button_layout.addWidget(self.clear_button)
        bottom_button_layout.addWidget(self.exit_button)

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_path:
            try:
                self.result_text.append(f"Attempting to load dataset from: {file_path}")
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
                self.result_text.append(f"Error loading dataset: {e}")

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

    def generate_correlation_matrix(self):

        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        numeric_cols = self.dataset.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            QMessageBox.warning(self, "Warning", "No numeric columns found for generating correlation matrix!")
            return

        correlation_matrix = numeric_cols.corr()
        self.result_text.append("\n=== Correlation Matrix ===\n")
        self.result_text.append(correlation_matrix.to_string())

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png", bbox_inches='tight')
        plt.close()

        pixmap = QPixmap("correlation_matrix.png")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)

    def trainModel(self):
        X, y = self.preprocessData()
        if X is None or y is None:
            self.result_text.append("No dataset loaded or invalid dataset!\n")
            return

        # Macierz korelacji
        self.generate_correlation_matrix()

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

        # Wywołanie walidacji krzyżowej
        self.calculate_cross_validation(X, y)

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

        load_action = file_menu.addAction("Load Dataset")
        load_action.triggered.connect(self.loadDataset)

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
