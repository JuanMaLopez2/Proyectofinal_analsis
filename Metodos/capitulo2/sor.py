from metodos.capitulo1.base import RootFindingMethod
import numpy as np
from prettytable import PrettyTable 
from sympy import sympify, Symbol, lambdify

import numpy as np

def spectral_radius_sor(A, w):
    A = np.array(A, dtype=float)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    T_sor = np.linalg.inv(D - w*L) @ ((1 - w)*D + w*U)
    eigenvalues = np.linalg.eigvals(T_sor)
    return max(abs(eigenvalues))

def parse_function(self, func_str):
        x = Symbol('x')
        try:
            expr = sympify(func_str, evaluate=True)
            func = lambdify(x, expr, modules=["numpy", "math"])
            return func
        except Exception as e:  
            raise ValueError(f"Error al parsear la función: {e}")
def sor(A,b,x0,n,tol,w):
    table=PrettyTable()
    table.field_names=["Iteraciones","Vector","Tolerancia"]
    it=0
    t=tol+1
    table.add_rows([[it,x0,t]])
    while t>tol:
        it+=1
        x_nuevo=np.zeros(n)
        for j in range(n):
            for k in range(n):
                if(j==k):
                    continue
                if(j>k):
                    x_nuevo[j]+=(-1)*A[j][k]*x_nuevo[k]
                else:
                    x_nuevo[j]+=(-1)*A[j][k]*x0[k]
            x_nuevo[j]+=b[j]
            x_nuevo[j]/=A[j][j]
            x_nuevo[j]=w*x_nuevo[j]+(1-w)*x0[j]
        t=max(abs(x0-x_nuevo))
        x0=x_nuevo
        table.add_rows([[it,x0,t]])
    return table

class SOR(RootFindingMethod):
    def __init__(self):
        super().__init__()

    def solve(self, A, b, x0, tol, niter, w=1.5):
        """
        Implementa el método SOR (Successive Over-Relaxation) para resolver sistemas lineales.
        """
        try:
            n = len(b)
            table = sor(A, b, x0, n, tol, w)
            return None, table, None, None
        except Exception as e:
            raise ValueError(f"Error en el método SOR: {str(e)}")

# Asegurarse de que la clase esté disponible para importación
__all__ = ['SOR']