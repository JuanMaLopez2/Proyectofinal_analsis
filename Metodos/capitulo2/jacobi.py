from metodos.capitulo1.base import RootFindingMethod
import numpy as np
from prettytable import PrettyTable
from sympy import sympify, Symbol, lambdify

class Jacobi(RootFindingMethod):
    def __init__(self):
        super().__init__()

    def parse_function(self, func_str):
        x = Symbol('x')
        try:
            expr = sympify(func_str, evaluate=True)
            func = lambdify(x, expr, modules=["numpy", "math"])
            return func
        except Exception as e:
            raise ValueError(f"Error al parsear la función: {e}")
        
    def solve(self, A, b, x0, tol, niter):
        """
        Implementa el método de Jacobi para resolver sistemas lineales.
        
        Args:
            A: Matriz de coeficientes
            b: Vector de términos independientes
            x0: Vector inicial
            tol: Tolerancia para el error
            niter: Número máximo de iteraciones
            
        Returns:
            iterations: Lista de iteraciones
            table: Tabla HTML con los resultados
            graph: Gráfico de la convergencia
            report: Reporte del proceso
        """
        try:
            # Validar entradas
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
            x0 = np.array(x0, dtype=float)
            n = len(b)
            
            if A.shape[0] != A.shape[1]:
                raise ValueError("La matriz A debe ser cuadrada")
            if A.shape[0] != b.shape[0]:
                raise ValueError("Las dimensiones de A y b no coinciden")
            if A.shape[0] != x0.shape[0]:
                raise ValueError("Las dimensiones de A y x0 no coinciden")
            
            # Calcular y mostrar radio espectral
            rho = self.compute_spectral_radius(A)
            print(f"[INFO] Radio espectral de Jacobi: {rho:.6f}")
            
            # Inicializar variables
            iterations = []
            x = x0.copy()
            i = 0
            error = tol + 1
            
            # Primera iteración
            iterations.append([i, x.copy(), error])
            
            # Iteraciones principales
            while error > tol and i < niter:
                i += 1
                x_nuevo = np.zeros(n)
                
                for j in range(n):
                    suma = sum(A[j][k] * x[k] for k in range(n) if k != j)
                    x_nuevo[j] = (b[j] - suma) / A[j][j]
                
                error = max(abs(x - x_nuevo))
                x = x_nuevo.copy()
                iterations.append([i, x.copy(), error])
                
                if error <= tol:
                    break
            
            # Generar resultados
            table = self.generate_table(iterations, ['Iteración', 'Vector', 'Error'])
            graph = self.generate_convergence_graph(iterations)
            report = self.generate_report(iterations, A, b, rho)
            
            return iterations, table, graph, report
            
        except Exception as e:
            raise ValueError(f"Error en el método de Jacobi: {str(e)}")

    def compute_spectral_radius(self, A):
        """
        Calcula el radio espectral de la matriz de iteración de Jacobi.
        """
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        D_inv = np.linalg.inv(D)
        BJ = -D_inv @ (L + U)
        eigenvalues = np.linalg.eigvals(BJ)
        rho = max(abs(eigenvalues))
        return rho

    def generate_convergence_graph(self, iterations):
        """Genera un gráfico de la convergencia del método."""
        import matplotlib.pyplot as plt
        
        errors = [it[2] for it in iterations]
        plt.figure(figsize=(10, 6))
        plt.semilogy(errors, 'b-o')
        plt.grid(True)
        plt.title('Convergencia del Método de Jacobi')
        plt.xlabel('Iteración')
        plt.ylabel('Error (log)')
        
        return plt

    def generate_report(self, iterations, A, b, rho):
        """Genera un reporte del proceso."""
        report = "<div class='reporte-jacobi'>"
        report += "<h3>Reporte del Método de Jacobi</h3>"
        report += f"<p>Radio espectral estimado: {rho:.6f}</p>"
        
        if iterations[-1][2] <= 1e-6:
            report += "<p class='success'>El método convergió exitosamente.</p>"
        else:
            report += "<p class='warning'>El método no convergió en el número máximo de iteraciones.</p>"
        
        report += f"<p>Número de iteraciones: {len(iterations)-1}</p>"
        report += f"<p>Error final: {iterations[-1][2]:.2e}</p>"
        report += f"<p>Solución final: {iterations[-1][1]}</p>"
        report += "</div>"
        return report

# Asegurarse de que la clase esté disponible para importación
__all__ = ['Jacobi']
