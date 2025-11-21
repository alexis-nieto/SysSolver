from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QGridLayout, QSpinBox, QTextEdit,
                             QFrame, QComboBox, QMessageBox)
from PyQt6.QtCore import Qt
from fractions import Fraction
from methods import solve_system, format_number


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solucionador (Modo Fracciones/Precisión)")
        self.setGeometry(100, 100, 1000, 600)

        # --- widget principal ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_grid = QGridLayout(main_widget)
        main_grid.setContentsMargins(10, 10, 10, 10)
        main_grid.setSpacing(10)

        # --- SECCION 1: entrada de datos ---
        self.section1 = QWidget()
        section1_layout = QVBoxLayout(self.section1)

        input_layout = QHBoxLayout()
        self.rows_input = QSpinBox()
        self.rows_input.setRange(2, 100);
        self.rows_input.setValue(3)
        self.cols_input = QSpinBox()
        self.cols_input.setRange(3, 100);
        self.cols_input.setValue(4)

        input_layout.addWidget(QLabel("Ecuaciones (Filas):"))
        input_layout.addWidget(self.rows_input)
        input_layout.addWidget(QLabel("Variables + 1 (Columnas):"))
        input_layout.addWidget(self.cols_input)
        input_layout.addStretch()
        section1_layout.addLayout(input_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        section1_layout.addWidget(self.scroll_area)
        self.matrix_widget = QWidget()
        self.scroll_area.setWidget(self.matrix_widget)

        # --- SECCION 2: resultados ---
        self.section2 = QWidget()
        section2_layout = QVBoxLayout(self.section2)
        section2_layout.addWidget(QLabel("<b>Matriz Final (Fracciones):</b>"))

        self.section2_scroll = QScrollArea()
        self.section2_scroll.setWidgetResizable(True)
        section2_layout.addWidget(self.section2_scroll)
        self.section2_matrix_widget = QWidget()
        self.section2_scroll.setWidget(self.section2_matrix_widget)

        # --- SECCION 3: pasos de solucion ---
        self.section3 = QWidget()
        section3_layout = QHBoxLayout(self.section3)

        controls_layout = QVBoxLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Gauss", "Gauss-Jordan", "Cramer", "Matriz Inversa"])
        self.calc_button = QPushButton("Calcular Solución")
        self.calc_button.clicked.connect(self.on_calculate)
        self.calc_button.setMinimumHeight(40)

        controls_layout.addWidget(QLabel("Método:"))
        controls_layout.addWidget(self.method_combo)
        controls_layout.addWidget(self.calc_button)
        controls_layout.addStretch()

        self.procedure_area = QTextEdit()
        self.procedure_area.setReadOnly(True)
        self.procedure_area.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")

        section3_layout.addLayout(controls_layout, 1)
        section3_layout.addWidget(self.procedure_area, 4)

        main_grid.addWidget(self.section1, 0, 0)
        main_grid.addWidget(self.section2, 0, 1)
        main_grid.addWidget(self.section3, 1, 0, 1, 2)

        main_grid.setColumnStretch(0, 5)
        main_grid.setColumnStretch(1, 4)
        main_grid.setRowStretch(0, 6)
        main_grid.setRowStretch(1, 4)

        self.create_matrix()
        self.clear_result_matrix()

        self.rows_input.valueChanged.connect(self.create_matrix)
        self.cols_input.valueChanged.connect(self.create_matrix)

    def get_var_name(self, index):
        return f"x{index + 1}"

    def create_matrix(self):
        rows = self.rows_input.value()
        cols = self.cols_input.value()

        self.matrix_widget = QWidget()
        layout = QGridLayout(self.matrix_widget)
        layout.setSpacing(5)

        for j in range(cols - 1):
            layout.addWidget(QLabel(self.get_var_name(j), alignment=Qt.AlignmentFlag.AlignCenter), 0, j)
        layout.addWidget(QLabel("=", alignment=Qt.AlignmentFlag.AlignCenter), 0, cols - 1)

        self.cells = []
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                val = QLineEdit("0")
                val.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(val, i + 1, j)
                row_cells.append(val)
            self.cells.append(row_cells)

        layout.setRowStretch(rows + 1, 1)
        self.scroll_area.setWidget(self.matrix_widget)
        self.clear_result_matrix()

    def clear_result_matrix(self):
        lbl = QLabel("Resultados")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.section2_scroll.setWidget(lbl)

    def display_result_matrix(self, result_matrix):
        if not result_matrix: return self.clear_result_matrix()

        widget = QWidget()
        layout = QGridLayout(widget)
        rows = len(result_matrix)
        cols = len(result_matrix[0])

        for j in range(cols - 1):
            lbl = QLabel(self.get_var_name(j))
            lbl.setStyleSheet("font-weight: bold")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl, 0, j)
        layout.addWidget(QLabel("=", alignment=Qt.AlignmentFlag.AlignCenter), 0, cols - 1)

        for i in range(rows):
            for j in range(cols):
                val = result_matrix[i][j]

                # usamos la funcion nueva para dar formato
                txt = format_number(val)

                lbl = QLabel(txt)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setFrameShape(QFrame.Shape.Box)

                # ajustamos el ancho segun que tan largo sea el texto (por si la fraccion es grande)
                width = max(60, len(txt) * 9)
                lbl.setMinimumWidth(width)

                color = "#d4f0f0" if j == cols - 1 else "white"
                lbl.setStyleSheet(f"background-color: {color}; border: 1px solid #aaa;")
                layout.addWidget(lbl, i + 1, j)

        layout.setRowStretch(rows + 1, 1)
        self.section2_scroll.setWidget(widget)

    def get_matrix_values(self):
        mat = []
        try:
            for r in self.cells:
                row = []
                for c in r:
                    txt = c.text().strip() or "0"
                    # convertimos lo que escribe el usuario a fraccion
                    # asi pueden poner "1/2" o "0.5" y funciona igual
                    try:
                        val = Fraction(txt)
                    except ValueError:
                        # si falla al convertir a fraccion, probamos con float
                        val = Fraction(float(txt))
                    row.append(val)
                mat.append(row)
            return mat
        except ValueError:
            QMessageBox.warning(self, "Error", "Ingrese números válidos (ej: 5, 1.2, 3/4)")
            return None

    def on_calculate(self):
        mat = self.get_matrix_values()
        if not mat: return

        self.procedure_area.setText("Calculando...")
        self.clear_result_matrix()

        try:
            sol, proc, final_mat = solve_system(mat, self.method_combo.currentText())
            self.procedure_area.setText(proc)
            if final_mat: self.display_result_matrix(final_mat)
        except Exception as e:
            self.procedure_area.setText(f"Error: {e}")