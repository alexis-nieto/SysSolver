import numpy as np
from typing import List, Tuple


class SystemSolver:
    """Clase para resolver sistemas de ecuaciones lineales usando diferentes métodos."""

    def __init__(self, matrix: List[List[float]]):
        """
        Inicializa el solver con una matriz aumentada.

        Args:
            matrix: Matriz aumentada [A|b] donde A son los coeficientes y b los términos independientes
        """
        self.original_matrix = np.array(matrix, dtype=float)
        self.n_equations = self.original_matrix.shape[0]
        self.n_variables = self.original_matrix.shape[1] - 1

    def gauss(self) -> Tuple[List[float], str]:
        """
        Método de eliminación de Gauss.

        Returns:
            Tuple con la solución y el procedimiento detallado
        """
        matrix = self.original_matrix.copy()
        procedure = "=== MÉTODO DE GAUSS ===\n\n"
        procedure += "Matriz inicial:\n"
        procedure += self._matrix_to_string(matrix) + "\n\n"

        # Fase de eliminación hacia adelante
        procedure += "--- Eliminación hacia adelante ---\n\n"

        for k in range(self.n_equations - 1):
            # Pivoteo parcial
            max_row = k
            for i in range(k + 1, self.n_equations):
                if abs(matrix[i][k]) > abs(matrix[max_row][k]):
                    max_row = i

            if max_row != k:
                matrix[[k, max_row]] = matrix[[max_row, k]]
                procedure += f"Intercambio de fila {k + 1} con fila {max_row + 1}:\n"
                procedure += self._matrix_to_string(matrix) + "\n\n"

            # Verificar si el pivote es cero
            if abs(matrix[k][k]) < 1e-10:
                return [], "Error: Sistema sin solución única (pivote cero)\n"

            # Eliminación
            for i in range(k + 1, self.n_equations):
                if matrix[i][k] != 0:
                    factor = matrix[i][k] / matrix[k][k]
                    procedure += f"F{i + 1} = F{i + 1} - ({factor:.4f}) * F{k + 1}\n"
                    matrix[i] = matrix[i] - factor * matrix[k]
                    procedure += self._matrix_to_string(matrix) + "\n\n"

        # Sustitución hacia atrás
        procedure += "--- Sustitución hacia atrás ---\n\n"
        solution = np.zeros(self.n_variables)

        for i in range(self.n_equations - 1, -1, -1):
            if abs(matrix[i][i]) < 1e-10:
                return [], "Error: Sistema sin solución única\n"

            sum_val = matrix[i][-1]
            for j in range(i + 1, self.n_variables):
                sum_val -= matrix[i][j] * solution[j]

            solution[i] = sum_val / matrix[i][i]

            var_name = chr(120 + i) if i < 3 else chr(94 + i)  # x, y, z, a, b, c...
            procedure += f"{var_name} = {solution[i]:.6f}\n"

        procedure += "\n=== SOLUCIÓN ===\n"
        for i, val in enumerate(solution):
            var_name = chr(120 + i) if i < 3 else chr(94 + i)
            procedure += f"{var_name} = {val:.6f}\n"

        return solution.tolist(), procedure

    def gauss_jordan(self) -> Tuple[List[float], str]:
        """
        Método de Gauss-Jordan (eliminación completa).

        Returns:
            Tuple con la solución y el procedimiento detallado
        """
        matrix = self.original_matrix.copy()
        procedure = "=== MÉTODO DE GAUSS-JORDAN ===\n\n"
        procedure += "Matriz inicial:\n"
        procedure += self._matrix_to_string(matrix) + "\n\n"

        procedure += "--- Eliminación hacia forma escalonada reducida ---\n\n"

        for k in range(self.n_equations):
            # Pivoteo parcial
            max_row = k
            for i in range(k + 1, self.n_equations):
                if abs(matrix[i][k]) > abs(matrix[max_row][k]):
                    max_row = i

            if max_row != k:
                matrix[[k, max_row]] = matrix[[max_row, k]]
                procedure += f"Intercambio de fila {k + 1} con fila {max_row + 1}:\n"
                procedure += self._matrix_to_string(matrix) + "\n\n"

            # Verificar pivote
            if abs(matrix[k][k]) < 1e-10:
                return [], "Error: Sistema sin solución única (pivote cero)\n"

            # Normalizar la fila del pivote
            pivot = matrix[k][k]
            if pivot != 1:
                procedure += f"F{k + 1} = F{k + 1} / {pivot:.4f}\n"
                matrix[k] = matrix[k] / pivot
                procedure += self._matrix_to_string(matrix) + "\n\n"

            # Eliminar elementos arriba y abajo del pivote
            for i in range(self.n_equations):
                if i != k and matrix[i][k] != 0:
                    factor = matrix[i][k]
                    procedure += f"F{i + 1} = F{i + 1} - ({factor:.4f}) * F{k + 1}\n"
                    matrix[i] = matrix[i] - factor * matrix[k]
                    procedure += self._matrix_to_string(matrix) + "\n\n"

        # Extraer solución
        solution = matrix[:, -1]

        procedure += "=== SOLUCIÓN ===\n"
        for i, val in enumerate(solution):
            var_name = chr(120 + i) if i < 3 else chr(94 + i)
            procedure += f"{var_name} = {val:.6f}\n"

        return solution.tolist(), procedure

    def cramer(self) -> Tuple[List[float], str]:
        """
        Regla de Cramer.

        Returns:
            Tuple con la solución y el procedimiento detallado
        """
        if self.n_equations != self.n_variables:
            return [], "Error: La regla de Cramer requiere un sistema cuadrado (n ecuaciones, n variables)\n"

        procedure = "=== REGLA DE CRAMER ===\n\n"

        # Separar coeficientes y términos independientes
        A = self.original_matrix[:, :-1]
        b = self.original_matrix[:, -1]

        procedure += "Matriz de coeficientes A:\n"
        procedure += self._matrix_to_string(A) + "\n\n"

        # Calcular determinante de A
        det_A = np.linalg.det(A)
        procedure += f"Determinante de A: det(A) = {det_A:.6f}\n\n"

        if abs(det_A) < 1e-10:
            return [], "Error: Sistema sin solución única (determinante = 0)\n"

        solution = []

        # Calcular cada variable
        for i in range(self.n_variables):
            # Reemplazar columna i con vector b
            A_i = A.copy()
            A_i[:, i] = b

            det_A_i = np.linalg.det(A_i)

            var_name = chr(120 + i) if i < 3 else chr(94 + i)
            procedure += f"Para {var_name}, reemplazamos columna {i + 1}:\n"
            procedure += self._matrix_to_string(A_i) + "\n"
            procedure += f"det(A_{i + 1}) = {det_A_i:.6f}\n"
            procedure += f"{var_name} = det(A_{i + 1}) / det(A) = {det_A_i:.6f} / {det_A:.6f} = {det_A_i / det_A:.6f}\n\n"

            solution.append(det_A_i / det_A)

        procedure += "=== SOLUCIÓN ===\n"
        for i, val in enumerate(solution):
            var_name = chr(120 + i) if i < 3 else chr(94 + i)
            procedure += f"{var_name} = {val:.6f}\n"

        return solution, procedure

    def inverse_matrix(self) -> Tuple[List[float], str]:
        """
        Método de matriz inversa (A⁻¹ * b = x).

        Returns:
            Tuple con la solución y el procedimiento detallado
        """
        if self.n_equations != self.n_variables:
            return [], "Error: La matriz inversa requiere un sistema cuadrado (n ecuaciones, n variables)\n"

        procedure = "=== MÉTODO DE MATRIZ INVERSA ===\n\n"

        # Separar coeficientes y términos independientes
        A = self.original_matrix[:, :-1]
        b = self.original_matrix[:, -1]

        procedure += "Matriz de coeficientes A:\n"
        procedure += self._matrix_to_string(A) + "\n\n"

        procedure += "Vector de términos independientes b:\n"
        procedure += self._vector_to_string(b) + "\n\n"

        # Verificar si la matriz es invertible
        det_A = np.linalg.det(A)
        procedure += f"Determinante de A: det(A) = {det_A:.6f}\n\n"

        if abs(det_A) < 1e-10:
            return [], "Error: La matriz no es invertible (determinante = 0)\n"

        # Calcular matriz inversa
        try:
            A_inv = np.linalg.inv(A)
            procedure += "Matriz inversa A⁻¹:\n"
            procedure += self._matrix_to_string(A_inv) + "\n\n"
        except np.linalg.LinAlgError:
            return [], "Error: No se pudo calcular la matriz inversa\n"

        # Calcular solución: x = A⁻¹ * b
        solution = np.dot(A_inv, b)

        procedure += "Solución x = A⁻¹ * b:\n"
        procedure += self._vector_to_string(solution) + "\n\n"

        procedure += "=== SOLUCIÓN ===\n"
        for i, val in enumerate(solution):
            var_name = chr(120 + i) if i < 3 else chr(94 + i)
            procedure += f"{var_name} = {val:.6f}\n"

        return solution.tolist(), procedure

    def _matrix_to_string(self, matrix) -> str:
        """Convierte una matriz a string formateado."""
        result = ""
        for row in matrix:
            result += "[ "
            for val in row:
                result += f"{val:8.4f} "
            result += "]\n"
        return result

    def _vector_to_string(self, vector) -> str:
        """Convierte un vector a string formateado."""
        result = "[ "
        for val in vector:
            result += f"{val:8.4f} "
        result += "]"
        return result


def solve_system(matrix: List[List[float]], method: str) -> Tuple[List[float], str]:
    """
    Función principal para resolver un sistema de ecuaciones.

    Args:
        matrix: Matriz aumentada [A|b]
        method: Método a usar ("Gauss", "Gauss-Jordan", "Cramer", "Matriz Inversa")

    Returns:
        Tuple con la solución y el procedimiento detallado
    """
    solver = SystemSolver(matrix)

    if method == "Gauss":
        return solver.gauss()
    elif method == "Gauss-Jordan":
        return solver.gauss_jordan()
    elif method == "Cramer":
        return solver.cramer()
    elif method == "Matriz Inversa":
        return solver.inverse_matrix()
    else:
        return [], f"Error: Método '{method}' no reconocido\n"