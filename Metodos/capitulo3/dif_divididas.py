import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def diferencias_divididas(x, y):
    """
    Calcula las diferencias divididas para el método de Newton.
    
    Args:
        x (np.array): Valores x
        y (np.array): Valores y
        
    Returns:
        np.array: Matriz de diferencias divididas
    """
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y
    
    for j in range(1, n):
        for i in range(n-j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x[i+j] - x[i])
    
    return F

def interpolacion_newton(x_values, y_values):
    """
    Realiza interpolación polinomial usando el método de Newton.
    
    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y
        
    Returns:
        tuple: (polinomio, expresión, gráfica)
    """
    # Convertir a arrays de numpy
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    # Calcular diferencias divididas
    F = diferencias_divididas(x, y)
    
    # Construir polinomio
    n = len(x)
    p = np.poly1d([0])
    
    for i in range(n):
        term = np.poly1d([1])
        for j in range(i):
            term = np.polymul(term, np.poly1d([1, -x[j]]))
        p = np.polyadd(p, F[0, i] * term)
    
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
    plt.title('Interpolación de Newton')
    
    # Convertir gráfica a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graph = base64.b64encode(image_png).decode('utf-8')
    
    return p, expr, graph 