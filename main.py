import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QGridLayout, QSpinBox)

from PyQt6.QtCore import Qt


class MatrixCreator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Creator")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Input section
        input_layout = QHBoxLayout()

        # Rows input
        rows_label = QLabel("Rows (n):")
        self.rows_input = QSpinBox()
        self.rows_input.setMinimum(2)
        self.rows_input.setMaximum(100)
        self.rows_input.setValue(3)

        # Columns input
        cols_label = QLabel("Columns (m):")
        self.cols_input = QSpinBox()
        self.cols_input.setMinimum(3)
        self.cols_input.setMaximum(100)
        self.cols_input.setValue(3)

        input_layout.addWidget(rows_label)
        input_layout.addWidget(self.rows_input)
        input_layout.addWidget(cols_label)
        input_layout.addWidget(self.cols_input)
        input_layout.addStretch()

        main_layout.addLayout(input_layout)

        # Scrollable matrix area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        main_layout.addWidget(self.scroll_area)

        # Matrix widget (initially empty)
        self.matrix_widget = QWidget()
        self.scroll_area.setWidget(self.matrix_widget)

        # Initial matrix creation
        self.create_matrix()

        # Connect spin boxes to create matrix automatically (after initial creation)
        self.rows_input.valueChanged.connect(self.create_matrix)
        self.cols_input.valueChanged.connect(self.create_matrix)

    def create_matrix(self):
        rows = self.rows_input.value()
        cols = self.cols_input.value()

        # Create new matrix widget
        self.matrix_widget = QWidget()
        # Reduce overall spacing so column titles sit closer to input fields
        matrix_layout = QGridLayout(self.matrix_widget)
        matrix_layout.setSpacing(2)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.setVerticalSpacing(0)

        # Variable labels (x, y, z, ..., w)
        variables = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w']

        # Add variable labels at the top (skip last column as it's the result)
        for j in range(cols - 1):
            if j < len(variables):
                var_label = QLabel(variables[j])
                var_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # Remove extra default margins/padding and reduce height so title is closer to the inputs
                var_label.setStyleSheet("font-weight: bold; padding: 0px; margin: 0px;")
                var_label.setFixedHeight(18)
                matrix_layout.addWidget(var_label, 0, j)

        # Add "Result" label for last column
        result_label = QLabel("=")
        result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_label.setStyleSheet("font-weight: bold; padding: 0px; margin: 0px;")
        result_label.setFixedHeight(18)
        matrix_layout.addWidget(result_label, 0, cols - 1)

        # Create matrix cells (starting from row 1 since row 0 has labels)
        self.cells = []
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                cell = QLineEdit()
                cell.setFixedWidth(60)
                # Slightly reduce height to better match label height
                cell.setFixedHeight(26)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setText("0")
                # Remove extra margins/padding inside the widget
                cell.setStyleSheet("padding: 2px; margin: 0px;")
                matrix_layout.addWidget(cell, i + 1, j)  # i + 1 to account for label row
                row_cells.append(cell)
            self.cells.append(row_cells)

        # Set the new widget to scroll area
        self.scroll_area.setWidget(self.matrix_widget)


def main():
    app = QApplication(sys.argv)
    window = MatrixCreator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()