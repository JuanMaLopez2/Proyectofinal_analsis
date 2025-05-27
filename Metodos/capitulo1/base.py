import numpy as np
import sympy as sp
from typing import Dict, Any, List, Union, Tuple
import matplotlib.pyplot as plt
import io
import urllib.parse
import base64

class RootFindingMethod:
    def __init__(self):
        self.x = sp.symbols('x')
        # Definir constantes matemáticas
        self.constants = {
            'e': sp.E,
            'pi': sp.pi,
            'i': sp.I,
            'inf': sp.oo,
            'nan': sp.nan
        }
        # Definir funciones matemáticas
        self.functions = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'log': sp.log,
            'ln': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'gamma': sp.gamma,
            'zeta': sp.zeta,
            'erf': sp.erf,
            'erfc': sp.erfc,
            're': sp.re,
            'im': sp.im,
            'conjugate': sp.conjugate,
            'arg': sp.arg
        }

    def parse_function(self, func_str: str) -> sp.Expr:
        """
        Parsea una función matemática en formato string a una expresión simbólica.
        Soporta números complejos y funciones matemáticas avanzadas.
        
        Args:
            func_str: String que representa la función matemática
            
        Returns:
            sp.Expr: Expresión simbólica de la función
        """
        try:
            # Reemplazar constantes y funciones por sus equivalentes simbólicos
            for const_name, const_value in self.constants.items():
                func_str = func_str.replace(const_name, str(const_value))
            
            for func_name, func_value in self.functions.items():
                func_str = func_str.replace(func_name, func_value.__name__)
            
            # Convertir a expresión simbólica
            expr = sp.sympify(func_str)
            return expr
        except Exception as e:
            raise ValueError(f"Error al parsear la función: {str(e)}")

    def evaluate_function(self, expr: sp.Expr, x_val: Union[float, complex]) -> Union[float, complex]:
        """
        Evalúa una función en un punto dado, soportando números complejos.
        
        Args:
            expr: Expresión simbólica de la función
            x_val: Valor de x (puede ser real o complejo)
            
        Returns:
            Union[float, complex]: Valor de la función evaluada
        """
        try:
            # Convertir el valor de x a complejo si es necesario
            if isinstance(x_val, (int, float)):
                x_val = complex(x_val)
            
            # Evaluar la expresión
            result = expr.subs(self.x, x_val)
            
            # Convertir el resultado a complejo si es necesario
            if isinstance(result, sp.Expr):
                result = complex(result)
            
            return result
        except Exception as e:
            raise ValueError(f"Error al evaluar la función: {str(e)}")

    def validate_inputs(self, **kwargs) -> None:
        """
        Valida los parámetros de entrada para los métodos.
        Incluye validaciones específicas para números complejos.
        
        Args:
            **kwargs: Parámetros a validar
        """
        # Validar tolerancia
        if 'tol' in kwargs and kwargs['tol'] <= 0:
            raise ValueError("La tolerancia debe ser positiva")
        
        # Validar número de iteraciones
        if 'niter' in kwargs and kwargs['niter'] <= 0:
            raise ValueError("El número de iteraciones debe ser positivo")
        
        # Validar intervalo [a,b]
        if 'a' in kwargs and 'b' in kwargs:
            if isinstance(kwargs['a'], (int, float)) and isinstance(kwargs['b'], (int, float)):
                if kwargs['a'] >= kwargs['b']:
                    raise ValueError("El límite inferior debe ser menor que el límite superior")
        
        # Validar valor inicial
        if 'x0' in kwargs:
            if not isinstance(kwargs['x0'], (int, float, complex)):
                raise ValueError("El valor inicial debe ser un número real o complejo")

    def generate_table(self, iterations: List[List[Any]]) -> str:
        """
        Genera una tabla HTML con los resultados de las iteraciones.
        Incluye soporte para números complejos.
        
        Args:
            iterations: Lista de iteraciones con sus resultados
            
        Returns:
            str: Tabla HTML con los resultados
        """
        table = "<table class='table table-striped'>"
        table += "<thead><tr><th>Iteración</th><th>Xi</th><th>F(Xi)</th><th>Error</th></tr></thead>"
        table += "<tbody>"
        
        for row in iterations:
            # Formatear números complejos
            xi = f"{row[1]:.6f}" if isinstance(row[1], (int, float)) else f"{complex(row[1]):.6f}"
            fxi = f"{row[2]:.6f}" if isinstance(row[2], (int, float)) else f"{complex(row[2]):.6f}"
            error = f"{row[3]:.6e}" if isinstance(row[3], (int, float)) else f"{complex(row[3]):.6e}"
            
            table += f"<tr><td>{row[0]}</td><td>{xi}</td><td>{fxi}</td><td>{error}</td></tr>"
        
        table += "</tbody></table>"
        return table

    def plot_function(self, expr: sp.Expr, iterations: List[List[Any]], 
                     x_range: Tuple[float, float] = (-10, 10)) -> str:
        """
        Genera una gráfica de la función y las iteraciones.
        Incluye soporte para funciones complejas.
        
        Args:
            expr: Expresión simbólica de la función
            iterations: Lista de iteraciones
            x_range: Rango de valores de x para la gráfica
            
        Returns:
            str: URI de la imagen en base64
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfica de la parte real
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        y_vals = [float(sp.re(expr.subs(self.x, val))) for val in x_vals]
        ax1.plot(x_vals, y_vals, label='Parte Real')
        ax1.set_title('Parte Real de la Función')
        ax1.grid(True)
        ax1.legend()
        
        # Gráfica de la parte imaginaria
        y_vals_im = [float(sp.im(expr.subs(self.x, val))) for val in x_vals]
        ax2.plot(x_vals, y_vals_im, label='Parte Imaginaria')
        ax2.set_title('Parte Imaginaria de la Función')
        ax2.grid(True)
        ax2.legend()
        
        # Marcar las iteraciones
        iter_x = [row[1] for row in iterations]
        iter_y_real = [float(sp.re(expr.subs(self.x, val))) for val in iter_x]
        iter_y_im = [float(sp.im(expr.subs(self.x, val))) for val in iter_x]
        
        ax1.plot(iter_x, iter_y_real, 'ro', label='Iteraciones')
        ax2.plot(iter_x, iter_y_im, 'ro', label='Iteraciones')
        
        # Guardar la figura
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        plt.close()
        
        return uri

    def generate_report(self, iterations: List[List[Any]], method_name: str) -> str:
        """
        Genera un reporte con los resultados del método.
        Incluye análisis para números complejos.
        
        Args:
            iterations: Lista de iteraciones
            method_name: Nombre del método
            
        Returns:
            str: Reporte en formato HTML
        """
        if not iterations:
            return "<p>No se encontraron resultados.</p>"
        
        report = f"<h3>Reporte del Método {method_name}</h3>"
        
        # Información general
        report += "<h4>Información General</h4>"
        report += f"<p>Número de iteraciones realizadas: {len(iterations)}</p>"
        
        # Resultado final
        final_iter = iterations[-1]
        report += "<h4>Resultado Final</h4>"
        report += f"<p>Raíz aproximada: {complex(final_iter[1]):.6f}</p>"
        report += f"<p>Error final: {complex(final_iter[3]):.6e}</p>"
        
        # Análisis de convergencia
        report += "<h4>Análisis de Convergencia</h4>"
        if len(iterations) > 1:
            errors = [complex(row[3]) for row in iterations[1:]]
            error_ratio = [abs(errors[i]/errors[i-1]) for i in range(1, len(errors))]
            avg_ratio = sum(error_ratio) / len(error_ratio)
            
            report += f"<p>Razón promedio de convergencia: {avg_ratio:.6f}</p>"
            if avg_ratio < 1:
                report += "<p>El método muestra convergencia lineal.</p>"
            else:
                report += "<p>El método muestra convergencia superlineal.</p>"
        
        return report 