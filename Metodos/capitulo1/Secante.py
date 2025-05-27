from .base import RootFindingMethod
import numpy as np
import sympy as sp
from sympy import sympify, Symbol, lambdify
class Secante(RootFindingMethod):
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
        
    def solve(self, x0, x1, tol, niter, fun):
        """
        Implementa el método de la secante para encontrar raíces.
        
        Args:
            x0: Primer valor inicial
            x1: Segundo valor inicial
            tol: Tolerancia para el error
            niter: Número máximo de iteraciones
            fun: Función f(x) en formato string
            
        Returns:
            iterations: Lista de iteraciones
            table: Tabla HTML con los resultados
            graph: Gráfico de la función y las iteraciones
            report: Reporte del proceso
        """
        try:
            # Validar entradas
            self.validate_inputs(x0, tol, niter)
            self.validate_inputs(x1, tol, niter)
            
            # Parsear función
            f = self.parse_function(fun)
            f_lambda = sp.lambdify(self.x, f, 'numpy')
            
            # Inicializar variables
            iterations = []
            xant = x0
            xact = x1
            i = 0
            error = tol + 1
            
            # Primera iteración
            fxant = f_lambda(xant)
            fxact = f_lambda(xact)
            iterations.append([i, xant, fxant, error])
            iterations.append([i+1, xact, fxact, error])
            
            # Iteraciones principales
            while error > tol and i < niter:
                i += 1
                if fxact == fxant:
                    raise ValueError("División por cero: f(x1) = f(x0)")
                
                xnext = xact - fxact * (xact - xant) / (fxact - fxant)
                error = abs(xnext - xact)
                fxnext = f_lambda(xnext)
                
                iterations.append([i+1, xnext, fxnext, error])
                
                xant = xact
                xact = xnext
                fxant = fxact
                fxact = fxnext
                
                if error <= tol:
                    break
            
            # Generar resultados
            table = self.generate_table(iterations, ['Iteración', 'Xi', 'f(Xi)', 'Error'])
            graph = self.generate_graph(f, iterations, x0)
            report = self.generate_report(iterations, fun)
            
            return iterations, table, graph, report
            
        except Exception as e:
            raise ValueError(f"Error en el método de la secante: {str(e)}")

# Asegurarse de que la clase esté disponible para importación
__all__ = ['Secante']

