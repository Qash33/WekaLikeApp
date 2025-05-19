from PyQt5.QtWidgets import QApplication
import sys
from gui.main_window import WekaLikeApp

# Punkt wejściowy do aplikacji.
# Tworzy QApplication, inicjalizuje i pokazuje główne okno.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WekaLikeApp()
    window.show()
    sys.exit(app.exec_())