from .base import RootFindingMethod
import numpy as np
from sympy import sympify, Symbol, lambdify


class ReglaFalsa(RootFindingMethod):

    def parse_function(self, func_str):
        x = Symbol('x')
        try:
            expr = sympify(func_str, evaluate=True)
            func = lambdify(x, expr, modules=["numpy", "math"])
            return func
        except Exception as e:
            raise ValueError(f"Error al parsear la función: {e}")
        
    def solve(self, a: float, b: float, tol: float, niter: int, fun: str) -> tuple:
        """
        Implementa el método de la regla falsa para encontrar raíces.
        
        Args:
            a: Límite inferior del intervalo
            b: Límite superior del intervalo
            tol: Tolerancia para el error
            niter: Número máximo de iteraciones
            fun: Función a evaluar en formato string
            
        Returns:
            tuple: (iteraciones, tabla, gráfica, reporte)
        """
        try:
            # Validar entradas
            self.validate_inputs(a=a, b=b, tol=tol, niter=niter)
            
            # Parsear la función
            expr = self.parse_function(fun)
            
            # Inicializar variables
            xi = a
            xs = b
            fxi = self.evaluate_function(expr, xi)
            fxs = self.evaluate_function(expr, xs)
            
            # Verificar si alguno de los extremos es raíz
            if abs(fxi) < tol:
                iterations = [[0, xi, fxi, 0]]
                return iterations, self.generate_table(iterations), \
                       self.plot_function(expr, iterations), \
                       self.generate_report(iterations, "Regla Falsa")
            
            if abs(fxs) < tol:
                iterations = [[0, xs, fxs, 0]]
                return iterations, self.generate_table(iterations), \
                       self.plot_function(expr, iterations), \
                       self.generate_report(iterations, "Regla Falsa")
            
            # Verificar cambio de signo
            if fxi * fxs > 0:
                raise ValueError("No hay cambio de signo en el intervalo")
            
            # Inicializar lista de iteraciones
            iterations = []
            
            # Método de la regla falsa
            xm = xi - (fxi * (xs - xi)) / (fxs - fxi)
            fxm = self.evaluate_function(expr, xm)
            i = 1
            error = tol + 1
            
            # Agregar primera iteración
            iterations.append([i, xm, fxm, error])
            
            # Iteraciones principales
            while error > tol and abs(fxm) > tol and i < niter:
                if fxi * fxm < 0:
                    xs = xm
                    fxs = fxm
                else:
                    xi = xm
                    fxi = fxm
                
                xaux = xm
                xm = xi - (fxi * (xs - xi)) / (fxs - fxi)
                fxm = self.evaluate_function(expr, xm)
                error = abs(xm - xaux)
                i += 1
                iterations.append([i, xm, fxm, error])
            
            # Verificar convergencia
            if error <= tol:
                mensaje = f"La raíz aproximada es {xm:.6f} con un error de {error:.2e}"
            elif abs(fxm) <= tol:
                mensaje = f"La raíz aproximada es {xm:.6f} con f(x) = {fxm:.2e}"
            else:
                mensaje = f"El método no convergió después de {niter} iteraciones"
            
            # Generar resultados
            table = self.generate_table(iterations)
            graph = self.plot_function(expr, iterations)
            report = self.generate_report(iterations, "Regla Falsa")
            
            return iterations, table, graph, report
            
        except Exception as e:
            raise ValueError(f"Error en el método de Regla Falsa: {str(e)}")

# Ejemplo de uso:
# f = lambda x: x**2 - 100
# resultado = ReglaF(5.0, 15.0, 1e-6, 100, f)
    
