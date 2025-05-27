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

def spline_cubico(x, y):
    """
    Calcula los coeficientes de los polinomios de interpolación de grado 3
    para un conjunto de n datos (x, y) usando el método de spline cúbico.
    
    Args:
        x: Vector de puntos x
        y: Vector de valores y
        
    Returns:
        tabla: Matriz con los coeficientes de los polinomios
    """
    # Validar puntos
    validar_puntos(x)
    
    n = len(x)
    tabla = np.zeros((n-1, 4))  # 4 coeficientes por polinomio (ax³ + bx² + cx + d)
    
    # Construir matriz A y vector b
    A = np.zeros((4*n-4, 4*n-4))
    b = np.zeros(4*n-4)
    
    # Condiciones de interpolación
    for i in range(n-1):
        A[4*i, 4*i] = x[i]**3
        A[4*i, 4*i+1] = x[i]**2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1
        A[4*i+1, 4*i] = x[i+1]**3
        A[4*i+1, 4*i+1] = x[i+1]**2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1
        b[4*i] = y[i]
        b[4*i+1] = y[i+1]
        
    # Condiciones de continuidad de derivadas
    for i in range(n-2):
        A[4*i+2, 4*i] = 3*x[i+1]**2
        A[4*i+2, 4*i+1] = 2*x[i+1]
        A[4*i+2, 4*i+2] = 1
        A[4*i+2, 4*i+4] = -3*x[i+1]**2
        A[4*i+2, 4*i+5] = -2*x[i+1]
        A[4*i+2, 4*i+6] = -1
        
        A[4*i+3, 4*i] = 6*x[i+1]
        A[4*i+3, 4*i+1] = 2
        A[4*i+3, 4*i+4] = -6*x[i+1]
        A[4*i+3, 4*i+5] = -2
        
    # Condiciones de frontera natural
    A[-2, 0] = 6*x[0]
    A[-2, 1] = 2
    A[-1, -4] = 6*x[-1]
    A[-1, -3] = 2
    
    # Resolver sistema
    try:
        coef = np.linalg.solve(A, b)
        for i in range(n-1):
            tabla[i, 0] = coef[4*i]  # Coeficiente cúbico
            tabla[i, 1] = coef[4*i+1]  # Coeficiente cuadrático
            tabla[i, 2] = coef[4*i+2]  # Coeficiente lineal
            tabla[i, 3] = coef[4*i+3]  # Término independiente
    except np.linalg.LinAlgError:
        raise ValueError("No se puede resolver el sistema para spline cúbico")
        
    return tabla

def evaluar_spline_cubico(x, tabla, x_val):
    """
    Evalúa el spline cúbico en un punto x_val.
    
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
            # Evaluar el polinomio cúbico
            coef = tabla[i]
            return coef[0]*x_val**3 + coef[1]*x_val**2 + coef[2]*x_val + coef[3]
            
    return None

def interpolacion_spline_cubico(x_values, y_values):
    """
    Realiza la interpolación por splines cúbicos y genera la gráfica.
    
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
            
        # Verificar que haya al menos 4 puntos para spline cúbico
        if len(x) < 4:
            raise ValueError("Se requieren al menos 4 puntos para spline cúbico")
        
        # Calcular coeficientes
        tabla = spline_cubico(x, y)
        
        # Generar puntos para la gráfica
        x_plot = np.linspace(min(x), max(x), 1000)
        y_plot = np.array([evaluar_spline_cubico(x, tabla, xi) for xi in x_plot])
        
        # Crear gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Puntos de datos')
        plt.plot(x_plot, y_plot, 'b-', label='Spline Cúbico')
        plt.grid(True)
        plt.legend()
        plt.title('Interpolación por Spline Cúbico')
        
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
            expr = f"{coef[0]:.6f}x³"
            if coef[1] >= 0:
                expr += f" + {coef[1]:.6f}x²"
            else:
                expr += f" - {abs(coef[1]):.6f}x²"
            if coef[2] >= 0:
                expr += f" + {coef[2]:.6f}x"
            else:
                expr += f" - {abs(coef[2]):.6f}x"
            if coef[3] >= 0:
                expr += f" + {coef[3]:.6f}"
            else:
                expr += f" - {abs(coef[3]):.6f}"
            polinomio.append(f"Para x ∈ [{x[i]}, {x[i+1]}]: {expr}")
            
        return "\n".join(polinomio), "Error calculado", graph
        
    except Exception as e:
        return str(e), "Error en el cálculo", None 