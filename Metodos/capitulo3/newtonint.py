import numpy as np

def newtonint(x, y):
    """
    Calcula los coeficientes del polinomio de interpolación de
    grado n-1 para el conjunto de n datos (x,y), mediante el método de Newton
    con diferencias divididas.
    
    Args:
        x (np.array): Valores x
        y (np.array): Valores y
        
    Returns:
        np.array: Tabla de diferencias divididas
    """
    n = len(x)
    # Crear tabla de n x (n+1) con ceros
    tabla = np.zeros((n, n+1))
    
    # Llenar primera columna con x y segunda con y
    tabla[:, 0] = x
    tabla[:, 1] = y
    
    # Calcular diferencias divididas
    for j in range(2, n+1):
        for i in range(j-1, n):
            tabla[i, j] = (tabla[i, j-1] - tabla[i-1, j-1]) / (tabla[i, 0] - tabla[i-j+1, 0])
    
    return tabla 