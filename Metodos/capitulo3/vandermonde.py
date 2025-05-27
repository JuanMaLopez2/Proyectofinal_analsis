import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def interpolacion_vandermonde(x_values, y_values):
    """
    Realiza interpolación polinomial usando el método de Vandermonde.
    
    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y
        
    Returns:
        tuple: (polinomio, error, gráfica)
    """
    # Convertir a arrays de numpy
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    # Crear matriz de Vandermonde
    V = np.vander(x, increasing=True)
    
    # Resolver sistema de ecuaciones
    try:
        coef = np.linalg.solve(V, y)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz de Vandermonde es singular. Verifique que los puntos x sean distintos.")
    
    # Crear función polinomial
    p = np.poly1d(coef[::-1])
    
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
    plt.title('Interpolación de Vandermonde')
    
    # Convertir gráfica a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graph = base64.b64encode(image_png).decode('utf-8')
    
    # Calcular error
    error = np.mean(np.abs(y - p(x)))
    
    return expr, f"{error:.6f}", graph 