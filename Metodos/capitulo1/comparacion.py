from typing import List, Dict, Any
import numpy as np
from .base import RootFindingMethod
from .biseccion import Biseccion
from .ReglaFalsa import ReglaFalsa
from .Puntofijo import PuntoFijo
from .Newton import Newton
from .Secante import Secante
from .RaicesMultiples import RaicesMultiples

class ComparadorMetodos:
    def __init__(self):
        self.metodos = {
            'Bisección': Biseccion(),
            'Regla Falsa': ReglaFalsa(),
            'Punto Fijo': PuntoFijo(),
            'Newton': Newton(),
            'Secante': Secante(),
            'Raíces Múltiples': RaicesMultiples()
        }
    
    def comparar_metodos(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara todos los métodos con los mismos parámetros y retorna un informe.
        
        Args:
            params: Diccionario con los parámetros comunes para todos los métodos
                   (x0, a, b, tol, niter, fun, etc.)
        
        Returns:
            Dict con los resultados de la comparación
        """
        resultados = {}
        
        for nombre, metodo in self.metodos.items():
            try:
                # Cada método puede requerir diferentes parámetros
                if nombre in ['Bisección', 'Regla Falsa']:
                    iterations, table, graph, report = metodo.solve(
                        params['a'], params['b'], params['tol'], 
                        params['niter'], params['fun']
                    )
                elif nombre == 'Punto Fijo':
                    iterations, table, graph, report = metodo.solve(
                        params['x0'], params['tol'], params['niter'],
                        params['fun'], params['g']
                    )
                elif nombre == 'Newton':
                    iterations, table, graph, report = metodo.solve(
                        params['x0'], params['tol'], params['niter'],
                        params['fun']
                    )
                elif nombre == 'Secante':
                    iterations, table, graph, report = metodo.solve(
                        params['x0'], params['x1'], params['tol'],
                        params['niter'], params['fun']
                    )
                elif nombre == 'Raíces Múltiples':
                    iterations, table, graph, report = metodo.solve(
                        params['x0'], params['tol'], params['niter'],
                        params['fun'], params['deriv1'], params['deriv2']
                    )
                
                resultados[nombre] = {
                    'iteraciones': len(iterations),
                    'error_final': iterations[-1][3] if iterations else float('inf'),
                    'raiz': iterations[-1][1] if iterations else None,
                    'converge': True
                }
                
            except Exception as e:
                resultados[nombre] = {
                    'iteraciones': 0,
                    'error_final': float('inf'),
                    'raiz': None,
                    'converge': False,
                    'error': str(e)
                }
        
        return self._generar_informe(resultados)
    
    def _generar_informe(self, resultados: Dict[str, Any]) -> str:
        """Genera un informe HTML con la comparación de métodos."""
        informe = "<div class='comparacion-metodos'>"
        informe += "<h3>Comparación de Métodos de Búsqueda de Raíces</h3>"
        
        # Tabla de comparación
        informe += "<table class='table table-striped'>"
        informe += "<thead><tr><th>Método</th><th>Iteraciones</th><th>Error Final</th><th>Raíz</th><th>Estado</th></tr></thead>"
        informe += "<tbody>"
        
        for nombre, datos in resultados.items():
            estado = "Converge" if datos['converge'] else f"No converge: {datos.get('error', 'Error desconocido')}"
            raiz = f"{datos['raiz']:.6f}" if datos['raiz'] is not None else "N/A"
            error = f"{datos['error_final']:.2e}" if datos['error_final'] != float('inf') else "N/A"
            
            informe += f"<tr><td>{nombre}</td><td>{datos['iteraciones']}</td><td>{error}</td><td>{raiz}</td><td>{estado}</td></tr>"
        
        informe += "</tbody></table>"
        
        # Análisis del mejor método
        metodos_convergentes = {k: v for k, v in resultados.items() if v['converge']}
        if metodos_convergentes:
            mejor_por_iteraciones = min(metodos_convergentes.items(), key=lambda x: x[1]['iteraciones'])
            mejor_por_error = min(metodos_convergentes.items(), key=lambda x: x[1]['error_final'])
            
            informe += "<div class='analisis'>"
            informe += "<h4>Análisis de Resultados</h4>"
            informe += f"<p>El método más eficiente en términos de iteraciones es <strong>{mejor_por_iteraciones[0]}</strong> "
            informe += f"con {mejor_por_iteraciones[1]['iteraciones']} iteraciones.</p>"
            informe += f"<p>El método más preciso es <strong>{mejor_por_error[0]}</strong> "
            informe += f"con un error final de {mejor_por_error[1]['error_final']:.2e}.</p>"
            informe += "</div>"
        else:
            informe += "<div class='alert alert-warning'>"
            informe += "<h4>No se encontraron métodos convergentes</h4>"
            informe += "<p>Ninguno de los métodos pudo converger con los parámetros dados.</p>"
            informe += "</div>"
        
        informe += "</div>"
        return informe 