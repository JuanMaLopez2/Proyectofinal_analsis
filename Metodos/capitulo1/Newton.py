import numpy as np
import sympy as sp

def newton(x0, tol, n, f):
    x = sp.Symbol('x')
    # Calcular la derivada simbólica
    df = sp.diff(f, x)
    # Convertir a funciones lambda
    f_lambda = sp.lambdify(x, f, 'numpy')
    df_lambda = sp.lambdify(x, df, 'numpy')
    
    xant = x0
    error = tol + 1
    cont = 0
    
    while error > tol and cont < n:
        if df_lambda(xant) == 0:
            return ["Error: derivada igual a cero"]
        xact = xant - f_lambda(xant)/df_lambda(xant)
        error = abs(xact - xant)
        cont += 1
        xant = xact
    
    if error <= tol:
        return [xact, f"Error de {error}"]
    else:
        return [f"Fracasó en {n} iteraciones"]

# Prueba del método
x = sp.Symbol('x')
print(newton(-10, 0.001, 5, ((x-4)**2)-100))