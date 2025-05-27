from .base import RootFindingMethod
import numpy as np
import sympy as sp
from sympy import sympify, Symbol, lambdify
class RaicesMultiples(RootFindingMethod):
    def __init__(self):
        super().__init__()
        self.x = sp.Symbol('x')
        self.f = (self.x)**2 - 100

    def parse_function(self, func_str):
        x = Symbol('x')
        try:
            expr = sympify(func_str, evaluate=True)
            func = lambdify(x, expr, modules=["numpy", "math"])
            return func
        except Exception as e:
            raise ValueError(f"Error al parsear la función: {e}")

    def solve(self, x0, tol, niter, fun, deriv1, deriv2):
        """
        Implementa el método de raíces múltiples para encontrar raíces.
        
        Args:
            x0: Valor inicial
            tol: Tolerancia para el error
            niter: Número máximo de iteraciones
            fun: Función f(x) en formato string
            deriv1: Primera derivada f'(x) en formato string
            deriv2: Segunda derivada f''(x) en formato string
            
        Returns:
            iterations: Lista de iteraciones
            table: Tabla HTML con los resultados
            graph: Gráfico de la función y las iteraciones
            report: Reporte del proceso
        """
        try:
            # Validar entradas
            self.validate_inputs(x0, tol, niter)
            
            # Parsear funciones
            f = self.parse_function(fun)
            derivada1 = self.parse_function(deriv1)
            derivada2 = self.parse_function(deriv2)
            
            # Convertir a funciones lambda
            f_lambda = sp.lambdify(self.x, f, 'numpy')
            fpx0 = sp.lambdify(self.x, derivada1, 'numpy')
            fppx0 = sp.lambdify(self.x, derivada2, 'numpy')
            
            # Inicializar variables
            iterations = []
            x0 = float(x0)
            fx0 = f_lambda(x0)
            
            if fx0 == 0:
                return [[0, x0, fx0, 0]], self.generate_table([[0, x0, fx0, 0]], ['Iteración', 'Xi', 'f(Xi)', 'Error']), self.generate_graph(f, [[0, x0, fx0, 0]], x0), self.generate_report([[0, x0, fx0, 0]], fun)
            
            i = 0
            error = tol + 1
            iterations.append([i, x0, fx0, error])
            
            while error > tol and fx0 != 0 and i < niter:
                fpx0_val = fpx0(x0)
                fppx0_val = fppx0(x0)
                den = (fpx0_val**2) - (fx0 * fppx0_val)
                
                if den == 0:
                    raise ValueError("División por cero en el denominador")
                
                x1 = x0 - ((fx0 * fpx0_val) / den)
                fx0 = f_lambda(x1)
                error = abs(x1 - x0)
                x0 = x1
                i += 1
                iterations.append([i, x0, fx0, error])
            
            if fx0 == 0:
                return iterations, self.generate_table(iterations, ['Iteración', 'Xi', 'f(Xi)', 'Error']), self.generate_graph(f, iterations, x0), self.generate_report(iterations, fun)
            elif error < tol:
                return iterations, self.generate_table(iterations, ['Iteración', 'Xi', 'f(Xi)', 'Error']), self.generate_graph(f, iterations, x0), self.generate_report(iterations, fun)
            else:
                raise ValueError(f"Fracasó en {niter} iteraciones")
                
        except Exception as e:
            raise ValueError(f"Error en el método de raíces múltiples: {str(e)}")

# Asegurarse de que la clase esté disponible para importación
__all__ = ['RaicesMultiples']

