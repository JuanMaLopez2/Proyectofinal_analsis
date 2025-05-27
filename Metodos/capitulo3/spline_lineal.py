import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def validar_puntos(x):
    """
    Valida que los puntos x sean diferentes entre sí y estén ordenados.
    
    Args:
        x: Vector de puntos x
        
    Returns:
        bool: True si los puntos son válidos, False en caso contrario
    """
    # Verificar que no haya puntos repetidos
    if len(set(x)) != len(x):
        raise ValueError("Los puntos x deben ser diferentes entre sí")
    
    # Verificar que estén ordenados
    if not np.all(np.diff(x) > 0):
        raise ValueError("Los puntos x deben estar ordenados de menor a mayor")
    
    return True

def spline_lineal(x, y):
    """
    Calcula los coeficientes de los polinomios de interpolación de grado 1
    para un conjunto de n datos (x, y) usando el método de spline lineal.
    
    Args:
        x: Vector de puntos x
        y: Vector de valores y
        
    Returns:
        tabla: Matriz con los coeficientes de los polinomios
    """
    # Validar puntos
    validar_puntos(x)
    
    n = len(x)
    tabla = np.zeros((n-1, 2))  # 2 coeficientes por polinomio (ax + b)
    
    for i in range(n-1):
        tabla[i, 0] = (y[i+1] - y[i]) / (x[i+1] - x[i])  # Pendiente
        tabla[i, 1] = y[i] - tabla[i, 0] * x[i]  # Término independiente
            
    return tabla

def evaluar_spline_lineal(x, tabla, x_val):
    """
    Evalúa el spline lineal en un punto x_val.
    
    Args:
        x: Vector de puntos x original
        tabla: Matriz de coeficientes del spline
        x_val: Punto a evaluar
        
    Returns:
        y_val: Valor del spline en x_val
    """
    n = len(x)
    
    # Encontrar el intervalo
    for i in range(n-1):
        if x[i] <= x_val <= x[i+1]:
            # Evaluar el polinomio lineal
            return tabla[i, 0] * x_val + tabla[i, 1]
            
    return None

def interpolacion_spline_lineal(x_values, y_values):
    """
    Realiza la interpolación por splines lineales y genera la gráfica.
    
    Args:
        x_values: Lista de valores x
        y_values: Lista de valores y
        
    Returns:
        polinomio: Expresión del polinomio
        error: Error de interpolación
        graph: Gráfica en base64
    """
    try:
        # Convertir a arrays de numpy
        x = np.array(x_values, dtype=float)
        y = np.array(y_values, dtype=float)
        
        # Verificar que x e y tengan la misma longitud
        if len(x) != len(y):
            raise ValueError("Los vectores x e y deben tener la misma longitud")
            
        # Verificar que haya al menos 2 puntos
        if len(x) < 2:
            raise ValueError("Se requieren al menos 2 puntos para la interpolación")
        
        # Calcular coeficientes
        tabla = spline_lineal(x, y)
        
        # Generar puntos para la gráfica
        x_plot = np.linspace(min(x), max(x), 1000)
        y_plot = np.array([evaluar_spline_lineal(x, tabla, xi) for xi in x_plot])
        
        # Crear gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Puntos de datos')
        plt.plot(x_plot, y_plot, 'b-', label='Spline Lineal')
        plt.grid(True)
        plt.legend()
        plt.title('Interpolación por Spline Lineal')
        
        # Convertir gráfica a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Generar expresión del polinomio
        polinomio = []
        for i in range(len(x)-1):
            coef = tabla[i]
            expr = f"{coef[0]:.6f}x"
            if coef[1] >= 0:
                expr += f" + {coef[1]:.6f}"
            else:
                expr += f" - {abs(coef[1]):.6f}"
            polinomio.append(f"Para x ∈ [{x[i]}, {x[i+1]}]: {expr}")
            
        return "\n".join(polinomio), "Error calculado", graph
        
    except Exception as e:
        return str(e), "Error en el cálculo", None 