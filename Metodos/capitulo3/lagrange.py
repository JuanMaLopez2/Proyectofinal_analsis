import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def lagrange(x, y):
    """
    Calcula los coeficientes del polinomio de interpolación de grado n-1
    para el conjunto de n datos (x,y), mediante el método de Lagrange.
    
    Args:
        x (list): Lista de valores x
        y (list): Lista de valores y
        
    Returns:
        np.poly1d: Polinomio interpolante
    """
    n = len(x)
    tabla = np.zeros((n, n))
    
    for i in range(n):
        Li = np.poly1d([1])  # Inicializar con 1
        den = 1
        
        for j in range(n):
            if j != i:
                paux = np.poly1d([1, -x[j]])  # [1 -x(j)] en MATLAB
                Li = np.polymul(Li, paux)     # conv en MATLAB
                den = den * (x[i] - x[j])
        
        tabla[i, :] = y[i] * Li.coef / den
    
    # Sumar todos los polinomios (equivalente a sum(Tabla) en MATLAB)
    pol = np.poly1d(np.sum(tabla, axis=0))
    return pol

def interpolacion_lagrange(x_values, y_values):
    """
    Realiza interpolación polinomial usando el método de Lagrange.
    
    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y
        
    Returns:
        tuple: (polinomio, expresión, gráfica)
    """
    # Convertir a arrays de numpy
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    # Calcular polinomio interpolante
    p = lagrange(x, y)
    
    # Generar expresión del polinomio
    expr = str(p)
    
    # Generar gráfica
    plt.figure(figsize=(10, 6))
    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = p(x_plot)
    
    plt.plot(x_plot, y_plot, 'b-', label='Polinomio interpolante')
    plt.plot(x, y, 'ro', label='Puntos de interpolación')
    plt.grid(True)
    plt.legend()
    plt.title('Interpolación de Lagrange')
    
    # Convertir gráfica a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graph = base64.b64encode(image_png).decode('utf-8')
    
    return p, expr, graph 