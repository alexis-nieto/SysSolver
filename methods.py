import numpy as np
from fractions import Fraction
from typing import List, Tuple, Any


def format_number(value: Any) -> str:
    """
    Formatea el número:
    - Si es Fraction y denominador 1 -> Entero
    - Si es Fraction -> Numerador/Denominador
    - Si es float -> Intenta convertir a fracción, sino 8 cifras significativas.
    """
    try:
        if isinstance(value, (int, np.integer)):
            return str(value)

        # Si ya es fracción
        if isinstance(value, Fraction):
            if value.denominator == 1:
                return str(value.numerator)
            return f"{value.numerator}/{value.denominator}"

        # Si es float, intentamos convertir a fracción exacta si es "bonita"
        # o usamos notación científica/decimal compacta si es compleja.
        val_float = float(value)

        # Verificar si es entero disfrazado de float (ej: 4.0)
        if val_float.is_integer():
            return str(int(val_float))

        # Intento de recuperar fracción de un float (útil para Cramer/Inversa que devuelven floats)
        try:
            frac = Fraction(val_float).limit_denominator(1000000)
            if abs(float(frac) - val_float) < 1e-9:
                if frac.denominator == 1: return str(frac.numerator)
                return f"{frac.numerator}/{frac.denominator}"
        except:
            pass

        # Fallback: 8 cifras significativas
        return f"{val_float:.8g}"

    except Exception:
        return str(value)


class SystemSolver:
    """Clase para resolver sistemas usando Fracciones exactas."""

    def __init__(self, matrix: List[List[Any]]):
        # Convertimos todo a Fraction al inicio
        # Usamos dtype=object para que numpy no convierta a float automáticamente
        self.original_matrix = np.array([
            [Fraction(x) for x in row] for row in matrix
        ], dtype=object)

        self.n_equations = self.original_matrix.shape[0]
        self.n_variables = self.original_matrix.shape[1] - 1

    def _get_variable_name(self, index: int) -> str:
        return f"x{index + 1}"

    def _get_solution_text(self, solution: List[Any]) -> str:
        text = "\n=== SOLUCIÓN ===\n"
        for i, val in enumerate(solution):
            var_name = self._get_variable_name(i)
            text += f"{var_name} = {format_number(val)}\n"
        return text

    def _create_identity_result_matrix(self, solution: List[Any]) -> List[List[Any]]:
        if not solution: return []
        # Matriz identidad de objetos (Fraction(0) y Fraction(1))
        final_mat = np.eye(self.n_equations, dtype=object)
        for r in range(self.n_equations):
            for c in range(self.n_equations):
                final_mat[r][c] = Fraction(1) if r == c else Fraction(0)

        sol_col = np.zeros(self.n_equations, dtype=object)
        for i in range(min(len(solution), self.n_equations)):
            sol_col[i] = solution[i]

        return np.column_stack((final_mat, sol_col)).tolist()

    def _matrix_to_string(self, matrix) -> str:
        result = ""
        # Calculamos ancho máximo para alinear bonito
        rows_str = []
        for row in matrix:
            rows_str.append([format_number(val) for val in row])

        # Alinear columnas
        if not rows_str: return ""
        col_widths = [max(len(row[i]) for row in rows_str) for i in range(len(rows_str[0]))]

        for row in rows_str:
            result += "[ "
            for i, val in enumerate(row):
                result += f"{val:>{col_widths[i]}} "
            result += "]\n"
        return result

    def gauss(self) -> Tuple[List[Any], str, List[List[Any]]]:
        matrix = self.original_matrix.copy()
        procedure = "=== MÉTODO DE GAUSS (Fracciones) ===\n\n"
        procedure += "Matriz inicial:\n" + self._matrix_to_string(matrix) + "\n"

        for k in range(self.n_equations - 1):
            max_row = k
            # Buscamos pivote (en valor absoluto)
            for i in range(k + 1, self.n_equations):
                if abs(matrix[i][k]) > abs(matrix[max_row][k]):
                    max_row = i

            if max_row != k:
                matrix[[k, max_row]] = matrix[[max_row, k]]
                procedure += f"Intercambio F{k + 1} <-> F{max_row + 1}:\n" + self._matrix_to_string(matrix) + "\n"

            if matrix[k][k] == 0:
                return [], procedure + "Error: Pivote cero.\n", []

            for i in range(k + 1, self.n_equations):
                if matrix[i][k] != 0:
                    factor = matrix[i][k] / matrix[k][k]
                    matrix[i] = matrix[i] - factor * matrix[k]
                    procedure += f"F{i + 1} = F{i + 1} - ({format_number(factor)}) * F{k + 1}\n"
                    procedure += self._matrix_to_string(matrix) + "\n"

        solution = np.zeros(self.n_variables, dtype=object)
        for i in range(self.n_equations - 1, -1, -1):
            if matrix[i][i] == 0:
                return [], procedure + "Error: Sistema sin solución única.\n", []

            sum_val = matrix[i][-1]
            for j in range(i + 1, self.n_variables):
                sum_val -= matrix[i][j] * solution[j]
            solution[i] = sum_val / matrix[i][i]

        procedure += self._get_solution_text(solution.tolist())
        return solution.tolist(), procedure, matrix.tolist()

    def gauss_jordan(self) -> Tuple[List[Any], str, List[List[Any]]]:
        matrix = self.original_matrix.copy()
        procedure = "=== MÉTODO DE GAUSS-JORDAN (Fracciones) ===\n\n"
        procedure += "Matriz inicial:\n" + self._matrix_to_string(matrix) + "\n"

        for k in range(self.n_equations):
            max_row = k
            for i in range(k + 1, self.n_equations):
                if abs(matrix[i][k]) > abs(matrix[max_row][k]):
                    max_row = i

            if max_row != k:
                matrix[[k, max_row]] = matrix[[max_row, k]]
                procedure += f"Intercambio F{k + 1} <-> F{max_row + 1}\n"

            if matrix[k][k] == 0:
                return [], procedure + "Error: Pivote cero.\n", []

            pivot = matrix[k][k]
            if pivot != 1:
                matrix[k] = matrix[k] / pivot
                procedure += f"F{k + 1} = F{k + 1} / {format_number(pivot)}\n"

            for i in range(self.n_equations):
                if i != k and matrix[i][k] != 0:
                    factor = matrix[i][k]
                    matrix[i] = matrix[i] - factor * matrix[k]

            procedure += self._matrix_to_string(matrix) + "\n"

        solution = matrix[:, -1]
        procedure += self._get_solution_text(solution.tolist())
        return solution.tolist(), procedure, matrix.tolist()

    def cramer(self) -> Tuple[List[Any], str, List[List[Any]]]:
        if self.n_equations != self.n_variables:
            return [], "Error: Requiere sistema cuadrado.\n", []

        procedure = "=== REGLA DE CRAMER ===\n\n"
        # Convertimos a float64 para calcular determinante si numpy se queja con objetos,
        # pero intentamos mantenerlo limpio.
        # Nota: Calcular det de Fractions con numpy es inestable, lo convertiremos a float para el cálculo
        # y luego convertiremos el resultado a fracción.

        A_obj = self.original_matrix[:, :-1]
        b_obj = self.original_matrix[:, -1]

        # Función auxiliar para determinante exacto (lento pero preciso) o fallback a float
        def get_det(mat):
            try:
                # Convertir a float para usar numpy det
                mat_float = np.array(mat, dtype=float)
                return np.linalg.det(mat_float)
            except:
                return 0.0

        det_A_val = get_det(A_obj)
        det_A_str = format_number(det_A_val)

        procedure += f"Matriz A:\n{self._matrix_to_string(A_obj)}\n"
        procedure += f"det(A) ≈ {det_A_str}\n\n"

        if abs(det_A_val) < 1e-10:
            return [], procedure + "Error: Determinante cero.\n", []

        solution = []
        for i in range(self.n_variables):
            A_i = A_obj.copy()
            A_i[:, i] = b_obj
            det_A_i_val = get_det(A_i)

            val = det_A_i_val / det_A_val
            # Intentar convertir resultado final a fracción
            val_frac = Fraction(val).limit_denominator(100000) if abs(val - round(val)) > 1e-10 else Fraction(
                int(round(val)))

            solution.append(val_frac)

            var_name = self._get_variable_name(i)
            procedure += f"Para {var_name}, det(A_{i + 1}) ≈ {format_number(det_A_i_val)}\n"
            procedure += f"{var_name} = {format_number(det_A_i_val)} / {det_A_str} = {format_number(val_frac)}\n\n"

        procedure += self._get_solution_text(solution)
        final_matrix = self._create_identity_result_matrix(solution)
        return solution, procedure, final_matrix

    def inverse_matrix(self) -> Tuple[List[Any], str, List[List[Any]]]:
        if self.n_equations != self.n_variables:
            return [], "Error: Requiere sistema cuadrado.\n", []

        procedure = "=== MATRIZ INVERSA ===\n\n"
        A = self.original_matrix[:, :-1]
        b = self.original_matrix[:, -1]

        # Numpy inv no funciona bien con dtype=object (Fractions).
        # Convertimos a float, calculamos y reconvertimos a fracción.
        try:
            A_float = np.array(A, dtype=float)
            A_inv_float = np.linalg.inv(A_float)
        except np.linalg.LinAlgError:
            return [], procedure + "Error: Matriz singular.\n", []

        # Reconvertir la inversa a fracciones para visualización
        A_inv_frac = []
        for r in range(self.n_equations):
            row_frac = []
            for c in range(self.n_equations):
                val = A_inv_float[r][c]
                # Heurística para encontrar la fracción
                f = Fraction(val).limit_denominator(100000)
                row_frac.append(f)
            A_inv_frac.append(row_frac)

        procedure += "Matriz Inversa A⁻¹ (Aproximada a Fracción):\n" + self._matrix_to_string(A_inv_frac) + "\n"

        solution_float = np.dot(A_inv_float, np.array(b, dtype=float))
        solution = [Fraction(x).limit_denominator(100000) for x in solution_float]

        procedure += "Solución x = A⁻¹ * b:\n"
        procedure += self._get_solution_text(solution)

        final_matrix = self._create_identity_result_matrix(solution)
        return solution, procedure, final_matrix


def solve_system(matrix, method):
    solver = SystemSolver(matrix)
    if method == "Gauss":
        return solver.gauss()
    elif method == "Gauss-Jordan":
        return solver.gauss_jordan()
    elif method == "Cramer":
        return solver.cramer()
    elif method == "Matriz Inversa":
        return solver.inverse_matrix()
    return [], "Método desconocido", []