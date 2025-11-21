import sys
from PyQt6.QtWidgets import QApplication
from gui import GUI  # importamos la clase de la interfaz

def initialize_gui():
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec())

def main():
    initialize_gui()

if __name__ == "__main__":
    main()