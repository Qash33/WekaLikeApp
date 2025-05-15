from PyQt5.QtWidgets import QApplication
import os
import sys
from gui.main_window import WekaLikeApp

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
# Punkt wejściowy do aplikacji.
# Tworzy QApplication, inicjalizuje i pokazuje główne okno.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WekaLikeApp()
    window.show()
    sys.exit(app.exec_())