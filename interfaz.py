import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
from Metodos.capitulo1 import ReglaFalsa, biseccion, Secante, RaicesMultiples, Puntofijo, Newton
from Metodos.capitulo2 import gaussSeidel, jacobi, sor, sorMatricial

class InterfazMetodos:
    def __init__(self, root):
        self.root = root
        self.root.title("Métodos Numéricos")
        self.root.geometry("800x600")
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True)
        
        # Pestaña para métodos de búsqueda de raíces
        self.tab_raices = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_raices, text="Métodos de Raíces")
        
        # Pestaña para métodos de sistemas lineales
        self.tab_sistemas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sistemas, text="Sistemas Lineales")
        
        self.setup_raices_tab()
        self.setup_sistemas_tab()
    
    def setup_raices_tab(self):
        # Frame para selección de método
        frame_metodo = ttk.LabelFrame(self.tab_raices, text="Selección de Método")
        frame_metodo.pack(pady=10, padx=10, fill="x")
        
        self.metodo_raices = ttk.Combobox(frame_metodo, values=[
            "Bisección", "Regla Falsa", "Punto Fijo", 
            "Newton", "Secante", "Raíces Múltiples"
        ])
        self.metodo_raices.pack(pady=5, padx=5)
        
        # Frame para parámetros
        frame_parametros = ttk.LabelFrame(self.tab_raices, text="Parámetros")
        frame_parametros.pack(pady=10, padx=10, fill="x")
        
        # Función
        ttk.Label(frame_parametros, text="Función (ej: x**2 - 4):").pack()
        self.funcion = ttk.Entry(frame_parametros)
        self.funcion.pack(pady=5)
        
        # Intervalo inicial
        ttk.Label(frame_parametros, text="Intervalo inicial (xi):").pack()
        self.xi = ttk.Entry(frame_parametros)
        self.xi.pack(pady=5)
        
        ttk.Label(frame_parametros, text="Intervalo final (xs):").pack()
        self.xs = ttk.Entry(frame_parametros)
        self.xs.pack(pady=5)
        
        # Tolerancia
        ttk.Label(frame_parametros, text="Tolerancia:").pack()
        self.tol = ttk.Entry(frame_parametros)
        self.tol.insert(0, "1e-6")
        self.tol.pack(pady=5)
        
        # Número máximo de iteraciones
        ttk.Label(frame_parametros, text="Número máximo de iteraciones:").pack()
        self.n = ttk.Entry(frame_parametros)
        self.n.insert(0, "100")
        self.n.pack(pady=5)
        
        # Botón de ejecución
        ttk.Button(self.tab_raices, text="Ejecutar", command=self.ejecutar_metodo_raices).pack(pady=10)
        
        # Área de resultados
        self.resultado_raices = tk.Text(self.tab_raices, height=10)
        self.resultado_raices.pack(pady=10, padx=10, fill="both", expand=True)
    
    def setup_sistemas_tab(self):
        # Frame para selección de método
        frame_metodo = ttk.LabelFrame(self.tab_sistemas, text="Selección de Método")
        frame_metodo.pack(pady=10, padx=10, fill="x")
        
        self.metodo_sistemas = ttk.Combobox(frame_metodo, values=[
            "Jacobi", "Gauss-Seidel", "SOR", "SOR Matricial"
        ])
        self.metodo_sistemas.pack(pady=5, padx=5)
        
        # Frame para matriz y vector
        frame_matriz = ttk.LabelFrame(self.tab_sistemas, text="Matriz y Vector")
        frame_matriz.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(frame_matriz, text="Matriz A (una fila por línea, elementos separados por espacios):").pack()
        self.matriz_a = tk.Text(frame_matriz, height=5)
        self.matriz_a.pack(pady=5)
        
        ttk.Label(frame_matriz, text="Vector b (elementos separados por espacios):").pack()
        self.vector_b = ttk.Entry(frame_matriz)
        self.vector_b.pack(pady=5)
        
        # Parámetros adicionales
        frame_parametros = ttk.LabelFrame(self.tab_sistemas, text="Parámetros")
        frame_parametros.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(frame_parametros, text="Tolerancia:").pack()
        self.tol_sistemas = ttk.Entry(frame_parametros)
        self.tol_sistemas.insert(0, "1e-6")
        self.tol_sistemas.pack(pady=5)
        
        ttk.Label(frame_parametros, text="Número máximo de iteraciones:").pack()
        self.n_sistemas = ttk.Entry(frame_parametros)
        self.n_sistemas.insert(0, "100")
        self.n_sistemas.pack(pady=5)
        
        # Factor de relajación para SOR
        ttk.Label(frame_parametros, text="Factor de relajación (w):").pack()
        self.w_sistemas = ttk.Entry(frame_parametros)
        self.w_sistemas.insert(0, "1.0")
        self.w_sistemas.pack(pady=5)
        
        # Botón de ejecución
        ttk.Button(self.tab_sistemas, text="Ejecutar", command=self.ejecutar_metodo_sistemas).pack(pady=10)
        
        # Área de resultados
        self.resultado_sistemas = tk.Text(self.tab_sistemas, height=10)
        self.resultado_sistemas.pack(pady=10, padx=10, fill="both", expand=True)
    
    def crear_funcion(self, f_str):
        x = sp.Symbol('x')
        try:
            # Convertir la expresión a una función sympy
            expr = sp.sympify(f_str)
            # Convertir la expresión a una función lambda
            f = sp.lambdify(x, expr, 'numpy')
            return f
        except Exception as e:
            raise ValueError(f"Error al procesar la función: {str(e)}")
    
    def ejecutar_metodo_raices(self):
        try:
            # Obtener parámetros
            metodo = self.metodo_raices.get()
            f_str = self.funcion.get()
            xi = float(self.xi.get())
            xs = float(self.xs.get())
            tol = float(self.tol.get())
            n = int(self.n.get())
            
            # Crear función usando sympy
            f = self.crear_funcion(f_str)
            
            # Ejecutar método seleccionado
            if metodo == "Bisección":
                resultado = biseccion.bisec(xi, xs, tol, n, f)
            elif metodo == "Regla Falsa":
                resultado = ReglaFalsa.ReglaF(xi, xs, tol, n, f)
            elif metodo == "Punto Fijo":
                resultado = Puntofijo.puntofijo(xi, tol, n, f)
            elif metodo == "Newton":
                resultado = Newton.newton(xi, tol, n, f)
            elif metodo == "Secante":
                resultado = Secante.secante(xi, xs, tol, n, f)
            elif metodo == "Raíces Múltiples":
                resultado = RaicesMultiples.raicesMultiples(xi, tol, n, f)
            
            # Mostrar resultado
            self.resultado_raices.delete(1.0, tk.END)
            self.resultado_raices.insert(tk.END, str(resultado))
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def ejecutar_metodo_sistemas(self):
        try:
            # Obtener parámetros
            metodo = self.metodo_sistemas.get()
            A_str = self.matriz_a.get("1.0", tk.END).strip()
            b_str = self.vector_b.get()
            tol = float(self.tol_sistemas.get())
            n = int(self.n_sistemas.get())
            w = float(self.w_sistemas.get())
            
            # Convertir strings a arrays numpy
            A = np.array([list(map(float, row.split())) for row in A_str.split('\n') if row.strip()])
            b = np.array(list(map(float, b_str.split())))
            
            # Ejecutar método seleccionado
            if metodo == "Jacobi":
                resultado = jacobi.jacobi(A, b, tol, n)
            elif metodo == "Gauss-Seidel":
                resultado = gaussSeidel.gaussSeidel(A, b, tol, n)
            elif metodo == "SOR":
                resultado = sor.sor(A, b, tol, n)
            elif metodo == "SOR Matricial":
                resultado = sorMatricial.sorMatricial(A, b, tol, n)
            
            # Mostrar resultado
            self.resultado_sistemas.delete(1.0, tk.END)
            self.resultado_sistemas.insert(tk.END, str(resultado))
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazMetodos(root)
    root.mainloop() 