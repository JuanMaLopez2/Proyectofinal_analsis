from .base import RootFindingMethod
from typing import List, Tuple
import numpy as np
from sympy import sympify, Symbol, lambdify

class Biseccion(RootFindingMethod):
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

    def solve(self, a: float, b: float, tol: float, niter: int, func_str: str) -> Tuple[List[List[float]], str, str, str]:
        # Validar entradas
        self.validate_inputs(None, a, b, tol, niter)
        
        # Parsear la función
        func = self.parse_function(func_str)
        
        # Evaluar la función en los extremos
        fa = self.evaluate_function(func, a)
        fb = self.evaluate_function(func, b)
        
        if fa * fb > 0:
            raise ValueError("La función no cambia de signo en el intervalo [a,b]")
        
        # Inicializar variables
        iterations = []
        c = 0
        error = tol + 1
        
        # Primera iteración
        xm = (a + b) / 2
        fm = self.evaluate_function(func, xm)
        iterations.append([c, xm, fm, error])
        
        # Iteraciones principales
        while error > tol and fm != 0 and c < niter:
            if fa * fm < 0:
                b = xm
                fb = fm
            else:
                a = xm
                fa = fm
            
            x_prev = xm
            xm = (a + b) / 2
            fm = self.evaluate_function(func, xm)
            error = abs(xm - x_prev)
            c += 1
            iterations.append([c, xm, fm, error])
        
        # Generar resultados
        table = self.generate_table(iterations)
        graph = self.plot_function(func, [a, b], [fa, fb], iterations)
        report = self.generate_report("Bisección", iterations, error, niter)
        
        return iterations, table, graph, report
