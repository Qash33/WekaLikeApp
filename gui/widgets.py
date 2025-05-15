from PyQt5.QtWidgets import QPushButton, QComboBox
from PyQt5.QtGui import QFont

def create_button(text, color, function):
    button = QPushButton(text)
    button.setFont(QFont("Arial", 12))
    button.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; border-radius: 5px;")
    button.clicked.connect(function)
    return button

def create_combo_box(items):
    combo = QComboBox()
    combo.setFont(QFont("Arial", 12))
    combo.addItems(items)
    return combo
