from .base import RootFindingMethod
import numpy as np
import sympy as sp
from sympy import sympify, Symbol, lambdify
class PuntoFijo(RootFindingMethod):
    def __init__(self):
        super().__init__()
        self.x = sp.Symbol("x")
        self.f = (self.x)**2 - 100

    def parse_function(self, func_str):
        x = Symbol('x')
        try:
            expr = sympify(func_str, evaluate=True)
            func = lambdify(x, expr, modules=["numpy", "math"])
            return func
        except Exception as e:
            raise ValueError(f"Error al parsear la función: {e}")
    def Cre_o_Decre(self, f, x0):
        return sp.diff(f, self.x).subs(self.x, x0) > 0

    def solve(self, x0, tol, niter, fun, g):
        """
        Implementa el método de punto fijo para encontrar raíces.
        
        Args:
            x0: Valor inicial
            tol: Tolerancia para el error
            niter: Número máximo de iteraciones
            fun: Función original f(x) en formato string
            g: Función de iteración g(x) en formato string
            
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
            g_func = self.parse_function(g)
            
            # Inicializar variables
            iterations = []
            xant = x0
            i = 0
            error = tol + 1
            
            # Primera iteración
            fx = self.evaluate_function(f, xant)
            iterations.append([i, xant, fx, error])
            
            # Iteraciones principales
            while error > tol and i < niter:
                i += 1
                xact = self.evaluate_function(g_func, xant)
                error = abs(xact - xant)
                fx = self.evaluate_function(f, xact)
                iterations.append([i, xact, fx, error])
                xant = xact
                
                if error <= tol:
                    break
            
            # Generar resultados
            table = self.generate_table(iterations, ['Iteración', 'Xi', 'f(Xi)', 'Error'])
            graph = self.generate_graph(f, iterations, x0)
            report = self.generate_report(iterations, fun, g)
            
            return iterations, table, graph, report
            
        except Exception as e:
            raise ValueError(f"Error en el método de punto fijo: {str(e)}")

# Asegurarse de que la clase esté disponible para importación
__all__ = ['PuntoFijo']