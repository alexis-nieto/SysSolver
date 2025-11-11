
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QGridLayout, QSpinBox, QTextEdit,
                             QFrame, QComboBox)
from PyQt6.QtCore import Qt


class MatrixCreator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solucionador de Sistemas de Ecuaciones")
        self.setGeometry(100, 100, 700, 350)

        # --- Main widget ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_grid = QGridLayout(main_widget)
        main_grid.setContentsMargins(8, 8, 8, 8)
        main_grid.setSpacing(8)

        # ---  top-left, section 1, data entry area ---
        self.section1 = QWidget()
        section1_layout = QVBoxLayout(self.section1)
        section1_layout.setContentsMargins(0, 0, 0, 0)
        section1_layout.setSpacing(6)

        # Input section (placed inside section1)
        input_layout = QHBoxLayout()

        # Rows input
        rows_label = QLabel("Ecuaciones (n):")
        self.rows_input = QSpinBox()
        self.rows_input.setMinimum(2)
        self.rows_input.setMaximum(100)
        self.rows_input.setValue(3)  # default to 3 equations

        # Columns input
        cols_label = QLabel("Columnas (m):")
        self.cols_input = QSpinBox()
        self.cols_input.setMinimum(3)
        self.cols_input.setMaximum(100)
        self.cols_input.setValue(4)  # default to 4 (3 variables + 1 result)

        input_layout.addWidget(rows_label)
        input_layout.addWidget(self.rows_input)
        input_layout.addWidget(cols_label)
        input_layout.addWidget(self.cols_input)
        input_layout.addStretch()

        section1_layout.addLayout(input_layout)

        # Scrollable matrix area (inside section1)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        section1_layout.addWidget(self.scroll_area)

        # Matrix widget (initially empty)
        self.matrix_widget = QWidget()
        self.scroll_area.setWidget(self.matrix_widget)

        # --- Section 2 (right/top) - mirror of section1 but read-only labels ---
        self.section2 = QWidget()
        section2_layout = QVBoxLayout(self.section2)
        section2_layout.setContentsMargins(0, 0, 0, 0)
        section2_layout.setSpacing(6)

        # Title for section2
        section2_layout.addWidget(QLabel("Matriz Resultante:"))

        # Scrollable matrix area for section2 (no size selectors here)
        self.section2_scroll = QScrollArea()
        self.section2_scroll.setWidgetResizable(True)
        self.section2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.section2_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        section2_layout.addWidget(self.section2_scroll)

        # Matrix widget placeholder for section2
        self.section2_matrix_widget = QWidget()
        self.section2_scroll.setWidget(self.section2_matrix_widget)

        # Container to keep label references for quick updates
        self.section2_labels = []

        # --- Section 3 (bottom) - spans both columns ---
        self.section3 = QWidget()
        section3_layout = QHBoxLayout(self.section3)
        section3_layout.setContentsMargins(0, 0, 0, 0)
        section3_layout.setSpacing(6)

        # Left controls: method selector + Calcular button
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        controls_layout.addWidget(QLabel("Metodo:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Gauss", "Gauss-Jordan", "Cramer", "Matriz Inversa"])
        controls_layout.addWidget(self.method_combo)

        self.calc_button = QPushButton("Calcular")
        self.calc_button.clicked.connect(self.on_calculate)
        controls_layout.addWidget(self.calc_button)

        controls_layout.addStretch()  # push controls to top

        section3_layout.addWidget(controls_widget)

        # Procedure text area (read-only) to the right of controls
        self.procedure_area = QTextEdit()
        self.procedure_area.setReadOnly(True)
        self.procedure_area.setPlaceholderText("El procedimiento aparecerá aquí...")
        self.procedure_area.setMinimumHeight(120)
        section3_layout.addWidget(self.procedure_area, stretch=1)

        # Add sections to main grid: top row has two columns, bottom row spans both
        main_grid.addWidget(self.section1, 0, 0)
        main_grid.addWidget(self.section2, 0, 1)
        main_grid.addWidget(self.section3, 1, 0, 1, 2)  # row 1, col 0, rowspan 1, colspan 2

        # Stretch factors so top row expands more and section2 gets reasonable width
        main_grid.setColumnStretch(0, 3)
        main_grid.setColumnStretch(1, 2)
        main_grid.setRowStretch(0, 9)
        main_grid.setRowStretch(1, 1)

        # Initial matrix creation
        self.create_matrix()

        # Connect spin boxes to recreate matrix automatically (section2 follows)
        self.rows_input.valueChanged.connect(self.create_matrix)
        self.cols_input.valueChanged.connect(self.create_matrix)

    def create_matrix(self):
        rows = self.rows_input.value()
        cols = self.cols_input.value()

        # Create new matrix widget (inside the scroll area already placed in section1)
        self.matrix_widget = QWidget()
        matrix_layout = QGridLayout(self.matrix_widget)
        matrix_layout.setSpacing(2)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.setVerticalSpacing(0)

        # Variable labels
        variables = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w']

        # Add variable labels at the top (skip last column as it's the result)
        for j in range(cols - 1):
            if j < len(variables):
                var_label = QLabel(variables[j])
                var_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
                cell.setFixedHeight(26)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setText("0")
                cell.setStyleSheet("padding: 2px; margin: 0px;")
                matrix_layout.addWidget(cell, i + 1, j)
                row_cells.append(cell)
            self.cells.append(row_cells)

        # Set the new matrix widget to the scroll area
        self.scroll_area.setWidget(self.matrix_widget)

        # After creating inputs, (re)create section2 labels and wire live updates
        self.update_section2_matrix()

        # Connect each input to update corresponding label in section2
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                cell = self.cells[i][j]
                cell.textChanged.connect(lambda text, i=i, j=j: self.on_cell_text_changed(i, j, text))

    def on_cell_text_changed(self, i, j, text):
        if i < len(self.section2_labels) and j < len(self.section2_labels[i]):
            self.section2_labels[i][j].setText(text)

    def update_section2_matrix(self):
        rows = self.rows_input.value()
        cols = self.cols_input.value()

        # Create new widget and layout for section2 matrix
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        # Same variable titles as section1
        variables = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w']

        # Column titles (top row)
        for j in range(cols - 1):
            if j < len(variables):
                var_label = QLabel(variables[j])
                var_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                var_label.setStyleSheet("font-weight: bold; padding: 0px; margin: 0px;")
                var_label.setFixedHeight(18)
                layout.addWidget(var_label, 0, j)

        result_label = QLabel("=")
        result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_label.setStyleSheet("font-weight: bold; padding: 0px; margin: 0px;")
        result_label.setFixedHeight(18)
        layout.addWidget(result_label, 0, cols - 1)

        # Cells as read-only labels (mirroring section1 size)
        self.section2_labels = []
        for i in range(rows):
            row_labels = []
            for j in range(cols):
                # If corresponding input exists, use its value; otherwise default "0"
                text = "0"
                if hasattr(self, "cells") and i < len(self.cells) and j < len(self.cells[i]):
                    text = self.cells[i][j].text()
                lbl = QLabel(text)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setFixedWidth(60)
                lbl.setFixedHeight(26)
                # give a framed look similar to input fields
                lbl.setFrameShape(QFrame.Shape.Box)
                lbl.setLineWidth(1)
                lbl.setStyleSheet("padding: 2px; margin: 0px;")
                layout.addWidget(lbl, i + 1, j)
                row_labels.append(lbl)
            self.section2_labels.append(row_labels)

        # Set constructed widget into the scroll area
        self.section2_matrix_widget = widget
        self.section2_scroll.setWidget(self.section2_matrix_widget)

    # Methods to control the procedure text area from within the app
    def set_procedure(self, text: str):
        """Replace procedure content."""
        self.procedure_area.setPlainText(text)

    def append_procedure(self, text: str):
        """Append text to the procedure area."""
        self.procedure_area.moveCursor(self.procedure_area.textCursor().End)
        self.procedure_area.insertPlainText(text)
        # keep scrollbar at bottom
        self.procedure_area.verticalScrollBar().setValue(self.procedure_area.verticalScrollBar().maximum())

    def on_calculate(self):
        """Triggered by the Calcular button. Replace stub with actual algorithm."""
        method = self.method_combo.currentText()
        # Example behavior: append a line stating the chosen method.
        self.append_procedure(f"Calculating using {method}...\n")
        # TODO: run chosen method on current matrix and append steps/results to procedure_area.


def main():
    app = QApplication(sys.argv)
    window = MatrixCreator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()