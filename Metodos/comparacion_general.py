from typing import Dict, Any, List
import numpy as np
from .capitulo1.ReglaFalsa import ReglaFalsa
from .capitulo1.biseccion import Biseccion
from .capitulo1.Puntofijo import PuntoFijo
from .capitulo1.Newton import Newton
from .capitulo1.Secante import Secante
from .capitulo1.RaicesMultiples import RaicesMultiples
from .capitulo2.jacobi import Jacobi
from .capitulo2.gaussSeidel import GaussSeidel
from .capitulo2.sor import SOR
from .capitulo2.sorMatricial import SORMatricial
from .capitulo3.vandermonde import interpolacion_vandermonde
from .capitulo3.newtonint import newtonint
from .capitulo3.lagrange import lagrange
from .capitulo3.spline_lineal import interpolacion_spline_lineal
from .capitulo3.spline_cubico import interpolacion_spline_cubico

class ComparadorGeneral:
    def __init__(self):
        self.metodos_cap1 = {
            'Bisección': Biseccion(),
            'Regla Falsa': ReglaFalsa(),
            'Punto Fijo': PuntoFijo(),
            'Newton': Newton(),
            'Secante': Secante(),
            'Raíces Múltiples': RaicesMultiples()
        }
        
        self.metodos_cap2 = {
            'Jacobi': Jacobi(),
            'Gauss-Seidel': GaussSeidel(),
            'SOR': SOR(),
            'SOR Matricial': SORMatricial()
        }
        
        self.metodos_cap3 = {
            'Vandermonde': interpolacion_vandermonde,
            'Newton Interpolante': newtonint,
            'Lagrange': lagrange,
            'Spline Lineal': interpolacion_spline_lineal,
            'Spline Cúbico': interpolacion_spline_cubico
        }
    
    def comparar_todos(self, resultados: Dict[str, Dict[str, Any]]) -> str:
        """
        Compara todos los métodos de los tres capítulos y genera un informe.
        
        Args:
            resultados: Diccionario con los resultados de cada método
                       {nombre_metodo: {iteraciones: int, error: float, ...}}
        
        Returns:
            str: Informe HTML con la comparación
        """
        informe = "<div class='comparacion-general'>"
        informe += "<h2>Comparación General de Métodos Numéricos</h2>"
        
        # Capítulo 1: Métodos de Búsqueda de Raíces
        informe += "<div class='capitulo'>"
        informe += "<h3>Capítulo 1: Métodos de Búsqueda de Raíces</h3>"
        informe += self._generar_tabla_capitulo(resultados, 'cap1')
        informe += "</div>"
        
        # Capítulo 2: Métodos de Sistemas Lineales
        informe += "<div class='capitulo'>"
        informe += "<h3>Capítulo 2: Métodos de Sistemas Lineales</h3>"
        informe += self._generar_tabla_capitulo(resultados, 'cap2')
        informe += "</div>"
        
        # Capítulo 3: Métodos de Interpolación
        informe += "<div class='capitulo'>"
        informe += "<h3>Capítulo 3: Métodos de Interpolación</h3>"
        informe += self._generar_tabla_capitulo(resultados, 'cap3')
        informe += "</div>"
        
        # Análisis General
        informe += "<div class='analisis-general'>"
        informe += "<h3>Análisis General</h3>"
        
        # Mejor método por capítulo
        for cap, nombre in [('cap1', 'Búsqueda de Raíces'), 
                          ('cap2', 'Sistemas Lineales'),
                          ('cap3', 'Interpolación')]:
            mejor = self._encontrar_mejor_metodo(resultados, cap)
            if mejor:
                informe += f"<p>En {nombre}, el mejor método es <strong>{mejor['nombre']}</strong> "
                informe += f"con {mejor['iteraciones']} iteraciones y un error de {mejor['error']:.2e}.</p>"
        
        # Mejor método general
        mejor_general = self._encontrar_mejor_metodo_general(resultados)
        if mejor_general:
            informe += "<h4>Mejor Método General</h4>"
            informe += f"<p>El mejor método en general es <strong>{mejor_general['nombre']}</strong> "
            informe += f"({mejor_general['capitulo']}) con {mejor_general['iteraciones']} iteraciones "
            informe += f"y un error de {mejor_general['error']:.2e}.</p>"
        
        informe += "</div>"
        informe += "</div>"
        
        return informe
    
    def _generar_tabla_capitulo(self, resultados: Dict[str, Dict[str, Any]], capitulo: str) -> str:
        """Genera una tabla HTML con los resultados de un capítulo específico."""
        tabla = "<table class='table table-striped'>"
        tabla += "<thead><tr><th>Método</th><th>Iteraciones</th><th>Error</th><th>Estado</th></tr></thead>"
        tabla += "<tbody>"
        
        for nombre, datos in resultados.items():
            if datos.get('capitulo') == capitulo:
                estado = "Converge" if datos.get('converge', True) else "No converge"
                error = f"{datos.get('error', float('inf')):.2e}" if datos.get('error') != float('inf') else "N/A"
                iteraciones = datos.get('iteraciones', 0)
                
                tabla += f"<tr><td>{nombre}</td><td>{iteraciones}</td><td>{error}</td><td>{estado}</td></tr>"
        
        tabla += "</tbody></table>"
        return tabla
    
    def _encontrar_mejor_metodo(self, resultados: Dict[str, Dict[str, Any]], capitulo: str) -> Dict[str, Any]:
        """Encuentra el mejor método de un capítulo específico."""
        metodos_cap = {k: v for k, v in resultados.items() if v.get('capitulo') == capitulo and v.get('converge', True)}
        if not metodos_cap:
            return None
        
        # Ponderación: 60% error, 40% iteraciones
        mejor = min(metodos_cap.items(), 
                   key=lambda x: 0.6 * x[1].get('error', float('inf')) + 
                                0.4 * x[1].get('iteraciones', float('inf')))
        
        return {
            'nombre': mejor[0],
            'iteraciones': mejor[1].get('iteraciones', 0),
            'error': mejor[1].get('error', float('inf'))
        }
    
    def _encontrar_mejor_metodo_general(self, resultados: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Encuentra el mejor método considerando todos los capítulos."""
        metodos_convergentes = {k: v for k, v in resultados.items() if v.get('converge', True)}
        if not metodos_convergentes:
            return None
        
        # Ponderación: 60% error, 40% iteraciones
        mejor = min(metodos_convergentes.items(),
                   key=lambda x: 0.6 * x[1].get('error', float('inf')) + 
                                0.4 * x[1].get('iteraciones', float('inf')))
        
        return {
            'nombre': mejor[0],
            'capitulo': mejor[1].get('capitulo', 'Desconocido'),
            'iteraciones': mejor[1].get('iteraciones', 0),
            'error': mejor[1].get('error', float('inf'))
        } 