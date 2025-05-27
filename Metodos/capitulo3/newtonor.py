import numpy as np

def newtonor(x, coef):
    """
    Calcula los coeficientes del polinomio simplificado de Newton,
    teniendo los coeficientes (coef) de la tabla de diferencias divididas y
    los puntos del conjunto de datos conocidos en la x.
    
    Args:
        x (np.array): Valores x
        coef (np.array): Coeficientes de la tabla de diferencias divididas
        
    Returns:
        np.poly1d: Polinomio interpolante de Newton
    """
    n = len(x)
    pol = np.poly1d([1])  # Inicializar con 1
    acum = pol
    pol = coef[0] * acum
    
    for i in range(n-1):
        # [0 pol] en MATLAB es equivalente a insertar un 0 al inicio
        pol = np.poly1d(np.insert(pol.coef, 0, 0))
        # conv(acum,[1 -x(i)]) en MATLAB es equivalente a polymul
        acum = np.polymul(acum, np.poly1d([1, -x[i]]))
        pol = pol + coef[i+1] * acum
    
    return pol 