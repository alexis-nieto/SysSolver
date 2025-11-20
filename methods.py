import numpy as np
from typing import List, Tuple


class SystemSolver:
    """Clase para resolver sistemas de ecuaciones lineales."""

    def __init__(self, matrix: List[List[float]]):
        self.original_matrix = np.array(matrix, dtype=float)
        self.n_equations = self.original_matrix.shape[0]
        self.n_variables = self.original_matrix.shape[1] - 1

    def _get_variable_name(self, index: int) -> str:
        """Genera nombres de variables consistentes con la GUI (x, y, z, a, b...)"""
        if index < 3:
            return chr(120 + index)  # x, y, z
        else:
            return chr(97 + index - 3)  # a, b, c...

    def _get_solution_text(self, solution: List[float]) -> str:
        """Genera el string final con los resultados."""
        text = "\n=== SOLUCIÓN ===\n"
        for i, val in enumerate(solution):
            var_name = self._get_variable_name(i)
            text += f"{var_name} = {val:.6f}\n"
        return text

    def _create_identity_result_matrix(self, solution: List[float]) -> List[List[float]]:
        """Crea una matriz visual para métodos que no resultan en una matriz final triangular (Cramer/Inversa)."""
        if not solution:
            return []
        final_mat = np.eye(self.n_equations)
        sol_col = np.zeros(self.n_equations)
        for i in range(min(len(solution), self.n_equations)):
            sol_col[i] = solution[i]
        return np.column_stack((final_mat, sol_col)).tolist()

    def _matrix_to_string(self, matrix) -> str:
        """Convierte matriz a texto para el log."""
        result = ""
        for row in matrix:
            result += "[ " + " ".join([f"{val:8.4f}" for val in row]) + " ]\n"
        return result

    def gauss(self) -> Tuple[List[float], str, List[List[float]]]:
        matrix = self.original_matrix.copy()
        procedure = "=== MÉTODO DE GAUSS ===\n\n"
        procedure += "Matriz inicial:\n" + self._matrix_to_string(matrix) + "\n\n"

        # Eliminación hacia adelante
        for k in range(self.n_equations - 1):
            # Pivoteo
            max_row = k
            for i in range(k + 1, self.n_equations):
                if abs(matrix[i][k]) > abs(matrix[max_row][k]):
                    max_row = i

            if max_row != k:
                matrix[[k, max_row]] = matrix[[max_row, k]]
                procedure += f"Intercambio F{k + 1} <-> F{max_row + 1}:\n" + self._matrix_to_string(matrix) + "\n"

            if abs(matrix[k][k]) < 1e-10:
                return [], procedure + "Error: Pivote cero.\n", []

            for i in range(k + 1, self.n_equations):
                if matrix[i][k] != 0:
                    factor = matrix[i][k] / matrix[k][k]
                    matrix[i] = matrix[i] - factor * matrix[k]
                    procedure += f"F{i + 1} = F{i + 1} - ({factor:.4f}) * F{k + 1}\n"
                    procedure += self._matrix_to_string(matrix) + "\n"

        # Sustitución hacia atrás
        solution = np.zeros(self.n_variables)
        for i in range(self.n_equations - 1, -1, -1):
            if abs(matrix[i][i]) < 1e-10:
                return [], procedure + "Error: Sistema sin solución única.\n", []

            sum_val = matrix[i][-1]
            for j in range(i + 1, self.n_variables):
                sum_val -= matrix[i][j] * solution[j]
            solution[i] = sum_val / matrix[i][i]

        # AQUÍ AGREGAMOS EL TEXTO DE LA SOLUCIÓN
        procedure += self._get_solution_text(solution.tolist())

        return solution.tolist(), procedure, matrix.tolist()

    def gauss_jordan(self) -> Tuple[List[float], str, List[List[float]]]:
        matrix = self.original_matrix.copy()
        procedure = "=== MÉTODO DE GAUSS-JORDAN ===\n\n"
        procedure += "Matriz inicial:\n" + self._matrix_to_string(matrix) + "\n\n"

        for k in range(self.n_equations):
            # Pivoteo
            max_row = k
            for i in range(k + 1, self.n_equations):
                if abs(matrix[i][k]) > abs(matrix[max_row][k]):
                    max_row = i

            if max_row != k:
                matrix[[k, max_row]] = matrix[[max_row, k]]
                procedure += f"Intercambio F{k + 1} <-> F{max_row + 1}\n"

            if abs(matrix[k][k]) < 1e-10:
                return [], procedure + "Error: Pivote cero.\n", []

            # Normalizar pivote a 1
            pivot = matrix[k][k]
            if pivot != 1:
                matrix[k] = matrix[k] / pivot
                procedure += f"F{k + 1} = F{k + 1} / {pivot:.4f}\n"

            # Hacer ceros arriba y abajo
            for i in range(self.n_equations):
                if i != k and matrix[i][k] != 0:
                    factor = matrix[i][k]
                    matrix[i] = matrix[i] - factor * matrix[k]

            procedure += self._matrix_to_string(matrix) + "\n"

        solution = matrix[:, -1]

        # AQUÍ AGREGAMOS EL TEXTO DE LA SOLUCIÓN
        procedure += self._get_solution_text(solution.tolist())

        return solution.tolist(), procedure, matrix.tolist()

    def cramer(self) -> Tuple[List[float], str, List[List[float]]]:
        if self.n_equations != self.n_variables:
            return [], "Error: Requiere sistema cuadrado.\n", []

        procedure = "=== REGLA DE CRAMER ===\n\n"
        A = self.original_matrix[:, :-1]
        b = self.original_matrix[:, -1]

        det_A = np.linalg.det(A)
        procedure += f"Matriz A:\n{self._matrix_to_string(A)}\n"
        procedure += f"det(A) = {det_A:.6f}\n\n"

        if abs(det_A) < 1e-10:
            return [], procedure + "Error: Determinante cero.\n", []

        solution = []
        for i in range(self.n_variables):
            A_i = A.copy()
            A_i[:, i] = b
            det_A_i = np.linalg.det(A_i)
            val = det_A_i / det_A
            solution.append(val)

            var_name = self._get_variable_name(i)
            procedure += f"Para {var_name}, det(A_{i + 1}) = {det_A_i:.4f}\n"
            procedure += f"{var_name} = {det_A_i:.4f} / {det_A:.4f} = {val:.6f}\n\n"

        # AQUÍ AGREGAMOS EL TEXTO DE LA SOLUCIÓN
        procedure += self._get_solution_text(solution)

        final_matrix = self._create_identity_result_matrix(solution)
        return solution, procedure, final_matrix

    def inverse_matrix(self) -> Tuple[List[float], str, List[List[float]]]:
        if self.n_equations != self.n_variables:
            return [], "Error: Requiere sistema cuadrado.\n", []

        procedure = "=== MATRIZ INVERSA ===\n\n"
        A = self.original_matrix[:, :-1]
        b = self.original_matrix[:, -1]

        try:
            A_inv = np.linalg.inv(A)
            procedure += "Matriz Inversa A⁻¹:\n" + self._matrix_to_string(A_inv) + "\n\n"
        except np.linalg.LinAlgError:
            return [], procedure + "Error: Matriz singular.\n", []

        solution = np.dot(A_inv, b)

        procedure += "Solución x = A⁻¹ * b:\n"
        # AQUÍ AGREGAMOS EL TEXTO DE LA SOLUCIÓN
        procedure += self._get_solution_text(solution.tolist())

        final_matrix = self._create_identity_result_matrix(solution.tolist())
        return solution.tolist(), procedure, final_matrix


def solve_system(matrix: List[List[float]], method: str) -> Tuple[List[float], str, List[List[float]]]:
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