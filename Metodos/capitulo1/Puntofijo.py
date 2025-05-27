from symtable import Symbol
import sympy
import numpy as np

x = sympy.Symbol("x")
f = (x)**2 - 100

def Cre_o_Decre(f,x0):
    return sympy.diff(f,x).subs(x,x0) > 0

def puntofijo(x0, tol, n, g):
    # Inicialización
    xant = x0
    error = tol + 1
    cont = 0
    
    # Ciclo
    while error > tol and cont < n:
        xact = g(xant)
        error = abs(xact - xant)
        cont += 1
        xant = xact
    
    if error <= tol:
        return [xact, f"Error de {error}"]
    else:
        return [f"Fracasó en {n} iteraciones"]

print(puntofijo(-12, 0.01, 100, lambda x: (x**2 - 100)**0.5))