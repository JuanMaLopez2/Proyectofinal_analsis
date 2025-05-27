import numpy as np
import matplotlib.pyplot as plt

def vandermonde(x, y):
    """
    Calcula el polinomio interpolante usando el método de Vandermonde
    y evalúa el error en puntos específicos.
    
    Args:
        x (np.array): Valores x
        y (np.array): Valores y
        
    Returns:
        tuple: (coeficientes, error1, error2, x_pol, y_pol, y_real)
    """
    # Crear matriz de Vandermonde
    A = np.column_stack([x**4, x**3, x**2, x, np.ones_like(x)])
    
    # Calcular coeficientes
    a = np.linalg.solve(A, y)
    
    # Puntos para graficar
    x_pol = np.arange(1.8, 5.01, 0.01)
    
    # Evaluar polinomio
    y_pol = a[0]*x_pol**4 + a[1]*x_pol**3 + a[2]*x_pol**2 + a[3]*x_pol + a[4]
    
    # Función real
    y_real = np.exp(-x_pol/1.8) + 1/(x_pol**2 - 3)
    
    # Calcular errores
    error1 = abs(a[0]*2.5**4 + a[1]*2.5**3 + a[2]*2.5**2 + a[3]*2.5 + a[4] - 
                 (np.exp(-2.5/1.8) + 1/(2.5**2 - 3)))
    
    error2 = abs(a[0]*6**4 + a[1]*6**3 + a[2]*6**2 + a[3]*6 + a[4] - 
                 (np.exp(-6/1.8) + 1/(6**2 - 3)))
    
    return a, error1, error2, x_pol, y_pol, y_real

def graficar_vandermonde(x, y, x_pol, y_pol, y_real):
    """
    Genera la gráfica del polinomio interpolante y la función real.
    
    Args:
        x (np.array): Valores x originales
        y (np.array): Valores y originales
        x_pol (np.array): Valores x para graficar
        y_pol (np.array): Valores y del polinomio
        y_real (np.array): Valores y de la función real
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r*', label='Puntos de interpolación')
    plt.plot(x_pol, y_pol, 'b-', label='Polinomio interpolante')
    plt.plot(x_pol, y_real, 'c--', label='Función real')
    plt.grid(True)
    plt.legend()
    plt.title('Interpolación de Vandermonde')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    x = np.array([1.8, 2, 3, 4, 5])
    y = np.exp(-x/1.8) + 1/(x**2 - 3)
    
    # Calcular interpolación
    a, error1, error2, x_pol, y_pol, y_real = vandermonde(x, y)
    
    # Mostrar resultados
    print("Coeficientes del polinomio:", a)
    print("Error en x = 2.5:", error1)
    print("Error en x = 6.0:", error2)
    
    # Graficar resultados
    graficar_vandermonde(x, y, x_pol, y_pol, y_real)