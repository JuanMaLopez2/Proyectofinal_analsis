from django.shortcuts import render, HttpResponse
from django import forms
from django.views.generic import TemplateView
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import io
import urllib, base64
from scipy.interpolate import lagrange, interp1d
from django.http import JsonResponse
from numpy.linalg import solve
from django.views.decorators.csrf import csrf_exempt
import json
from metodos.capitulo3.newtonint import newtonint
from metodos.capitulo3.newtonor import newtonor
from metodos.capitulo3.lagrange import lagrange
from metodos.capitulo3.spline_lineal import interpolacion_spline_lineal
from metodos.capitulo3.spline_cubico import interpolacion_spline_cubico
from metodos.capitulo3.vandermonde import interpolacion_vandermonde
from metodos.capitulo1.biseccion import Biseccion
from metodos.capitulo1.comparacion import ComparadorMetodos
from metodos.comparacion_general import ComparadorGeneral

#-------------Para ingresar todos los parametros de los metodos--------------------------
class FunctionForm(forms.Form):
    function = forms.CharField(label='Función', widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
#---------------------------------------------------------------------------------------
class HomePageView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = FunctionForm()
        return context

    def post(self, request, *args, **kwargs):
        form = FunctionForm(request.POST)
        if form.is_valid():
            function_str = form.cleaned_data['function']
            x = sp.symbols('x')
            try:
                # Convertir la cadena de texto a una expresión simbólica
                function_expr = sp.sympify(function_str)
                # Generar valores de x
                x_values = np.linspace(-10, 10, 400)
                # Evaluar la función para cada valor de x
                y_values = [function_expr.subs(x, val) for val in x_values]

                # Crear gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Función'))

                plot_div = plot(fig, output_type='div', include_plotlyjs=False)
                context = self.get_context_data()
                context['form'] = form
                context['plot_div'] = plot_div
                return render(request, self.template_name, context)
            except (sp.SympifyError, TypeError):
                form.add_error('function', 'Ingrese una función válida en términos de x.')
                context = self.get_context_data()
                context['form'] = form
                return render(request, self.template_name, context)
        else:
            context = self.get_context_data()
            context['form'] = form
            return render(request, self.template_name, context)

class MetodosPageView(TemplateView):
    template_name = 'metodos.html'

class NosotrosPageView(TemplateView):
    template_name = 'nosotros.html'


#-------------------------------------METODOS DESDE AQUI-------------------------------------------------------------------------
#----------BISECCION----------------------
class BiseccionForm(forms.Form):
    a = forms.FloatField(label='Límite inferior (a)', required=True)
    b = forms.FloatField(label='Límite superior (b)', required=True)
    tol = forms.FloatField(label='Tolerancia', required=True)
    niter = forms.IntegerField(label='Número máximo de iteraciones', required=True)
    fun = forms.CharField(label='Función f(x)', required=True, 
                         widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class BiseccionPageView(TemplateView):
    template_name = 'biseccion.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = BiseccionForm()
        return context

    def post(self, request, *args, **kwargs):
        form = BiseccionForm(request.POST)
        context = self.get_context_data()
        
        if form.is_valid():
            try:
                a = form.cleaned_data['a']
                b = form.cleaned_data['b']
                tol = form.cleaned_data['tol']
                niter = form.cleaned_data['niter']
                fun = form.cleaned_data['fun']
                export = form.cleaned_data['export']

                # Crear instancia del método de bisección
                biseccion = Biseccion()
                
                # Resolver el problema
                iterations, table, graph, report = biseccion.solve(a, b, tol, niter, fun)
                
                # Generar comparación automática
                comparador = ComparadorMetodos()
                params = {
                    'a': a,
                    'b': b,
                    'tol': tol,
                    'niter': niter,
                    'fun': fun,
                    'x0': (a + b) / 2,  # Valor inicial para métodos que lo requieren
                    'x1': b,  # Para el método de la secante
                    'g': fun,  # Para punto fijo
                    'deriv1': str(sp.diff(sp.sympify(fun), 'x')),  # Para Newton y Raíces Múltiples
                    'deriv2': str(sp.diff(sp.diff(sp.sympify(fun), 'x'), 'x'))  # Para Raíces Múltiples
                }
                comparacion = comparador.comparar_metodos(params)
                
                # Exportar resultados si se solicita
                if export:
                    response = HttpResponse(content_type='text/plain')
                    response['Content-Disposition'] = 'attachment; filename="resultados_biseccion.txt"'
                    response.write("Iteración\tXi\tF(Xi)\tError\n")
                    for row in iterations:
                        response.write("\t".join(map(str, row)) + "\n")
                    return response

                context['table'] = table
                context['graph'] = graph
                context['report'] = report
                context['comparacion'] = comparacion
                context['form'] = form
                
            except ValueError as e:
                context['error'] = str(e)
                context['form'] = form
            except Exception as e:
                context['error'] = f"Error inesperado: {str(e)}"
                context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#---------PUNTO_FIJO--------------------
# Formulario para el método de punto fijo
class PuntoFijoForm(forms.Form):
    x0 = forms.FloatField(label='Valor inicial X0', required=True)
    tol = forms.FloatField(label='Tolerancia', required=True)
    niter = forms.IntegerField(label='Número máximo de iteraciones', required=True)
    fun = forms.CharField(label='Función f(x)', required=True)
    g = forms.CharField(label='Función g(x)', required=True)
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class PuntoFijoPageView(TemplateView):
    template_name = 'puntofijo.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = PuntoFijoForm()
        return context

    def post(self, request, *args, **kwargs):
        form = PuntoFijoForm(request.POST)
        context = self.get_context_data()
        
        if form.is_valid():
            try:
                x0 = form.cleaned_data['x0']
                tol = form.cleaned_data['tol']
                niter = form.cleaned_data['niter']
                fun = form.cleaned_data['fun']
                g = form.cleaned_data['g']
                export = form.cleaned_data['export']

                # Crear instancia del método de punto fijo
                puntofijo = PuntoFijo()
                
                # Resolver el problema
                iterations, table, graph, report = puntofijo.solve(x0, tol, niter, fun, g)
                
                # Generar comparación automática
                comparador = ComparadorMetodos()
                params = {
                    'x0': x0,
                    'tol': tol,
                    'niter': niter,
                    'fun': fun,
                    'g': g,
                    'a': x0 - 1,  # Para métodos que requieren intervalo
                    'b': x0 + 1,  # Para métodos que requieren intervalo
                    'x1': x0 + 0.1,  # Para el método de la secante
                    'deriv1': str(sp.diff(sp.sympify(fun), 'x')),  # Para Newton y Raíces Múltiples
                    'deriv2': str(sp.diff(sp.diff(sp.sympify(fun), 'x'), 'x'))  # Para Raíces Múltiples
                }
                comparacion = comparador.comparar_metodos(params)
                
                # Agregar resultados al contexto
                context['iterations'] = iterations
                context['table'] = table
                context['graph'] = graph
                context['report'] = report
                context['comparacion'] = comparacion
                context['form'] = form
                
                if export:
                    # Generar archivo de exportación
                    response = HttpResponse(content_type='text/plain')
                    response['Content-Disposition'] = f'attachment; filename="puntofijo_resultados.txt"'
                    response.write(f"Método de Punto Fijo\n")
                    response.write(f"==================\n\n")
                    response.write(f"Función: {fun}\n")
                    response.write(f"Función de iteración: {g}\n")
                    response.write(f"Valor inicial: {x0}\n")
                    response.write(f"Tolerancia: {tol}\n")
                    response.write(f"Número máximo de iteraciones: {niter}\n\n")
                    response.write(f"Resultados:\n")
                    response.write(f"----------\n")
                    response.write(f"Raíz aproximada: {iterations[-1][1]}\n")
                    response.write(f"Error final: {iterations[-1][3]}\n")
                    response.write(f"Número de iteraciones: {len(iterations)}\n")
                    return response
                
            except Exception as e:
                context['error'] = str(e)
        
        return render(request, self.template_name, context)

#---------Regla_Falsa-------------------
class ReglaFalsaForm(forms.Form):
    xi = forms.FloatField(label='Xi', required=True)
    xs = forms.FloatField(label='Xs', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    niter = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class ReglaFalsaPageView(TemplateView):
    template_name = 'reglafalsa.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = ReglaFalsaForm()
        return context

    def post(self, request, *args, **kwargs):
        form = ReglaFalsaForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            xi = form.cleaned_data['xi']
            xs = form.cleaned_data['xs']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            fun = form.cleaned_data['fun']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            f = sp.sympify(fun)
            fi = f.subs(x, xi)
            fs = f.subs(x, xs)
            iterations = []
            iter_nums = []
            x_ants = []

            if fi * fs > 0:
                result = "El intervalo no es adecuado"
            else:
                xm = xi - (fi * (xs - xi) / (fs - fi))
                fxm = f.subs(x, xm)
                iter_nums.append(0)
                x_ants.append(float(xm))
                iterations.append([0, float(xi), float(xs), float(xm), float(fxm), None])
                error = tol + 1
                c = 1

                while error > tol and fxm != 0 and c < niter:
                    if fi * fxm < 0:
                        xs = xm
                        fs = f.subs(x, xs)
                    else:
                        xi = xm
                        fi = f.subs(x, xi)

                    x_prev = xm
                    xm = xi - (fi * (xs - xi) / (fs - fi))
                    fxm = f.subs(x, xm)
                    error = abs(xm - x_prev)
                    iter_nums.append(c)
                    x_ants.append(float(xm))
                    iterations.append([c, float(xi), float(xs), float(xm), float(fxm), float(error)])
                    c += 1

                if fxm == 0:
                    result = f"{xm} es una raíz exacta"
                elif error < tol:
                    result = f"{xm} es una aproximación de una raíz con una tolerancia {tol}"
                else:
                    result = f"El método fracasó en {niter} iteraciones"

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_regla_falsa.txt"'
                response.write(f"Resultado: {result}\n")
                response.write("Iteración\tXi\tXs\tXm\tf(Xm)\tError\n")
                for row in iterations:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            # Graficar la convergencia
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(iter_nums, x_ants, 'ro-', label='Convergencia del método de Regla Falsa')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Valores de $X_{ant}$')
            ax.set_title('Convergencia del método de Regla Falsa')
            ax.grid(True)
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
            context['image'] = uri

        else:
            context['form'] = form

        return render(request, self.template_name, context)

#----------NEWTON----------------------
class NewtonForm(forms.Form):
    x0 = forms.FloatField(label='Valor inicial X0', required=True)
    tol = forms.FloatField(label='Tolerancia', required=True)
    niter = forms.IntegerField(label='Número máximo de iteraciones', required=True)
    fun = forms.CharField(label='Función f(x)', required=True)
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class NewtonPageView(TemplateView):
    template_name = 'newton.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = NewtonForm()
        return context

    def post(self, request, *args, **kwargs):
        form = NewtonForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            f_str = form.cleaned_data['fun']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            f = sp.sympify(f_str)
            df = sp.diff(f, x)

            def newton(f, df, x0, tol, niter):
                iteraciones = []
                i = 0
                error = tol + 1
                iteraciones.append([i, x0, f.subs(x, x0), error])
                while i < niter and error > tol:
                    fx = f.subs(x, x0)
                    dfx = df.subs(x, x0)
                    if dfx == 0:
                        break
                    xn = x0 - fx / dfx
                    i += 1
                    error = abs(xn - x0)
                    iteraciones.append([i, xn, f.subs(x, xn), error])
                    x0 = xn
                return iteraciones

            iteraciones = newton(f, df, x0, tol, niter)
            result = iteraciones[-1][1] if iteraciones else []

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_newton.txt"'
                response.write("Iteración\tX0\tF(X0)\tTolerancia\n")
                for row in iteraciones:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
                return response

            # Graficar la función y las iteraciones
            x_vals = np.linspace(x0 - 2, x0 + 2, 400)
            y_vals = [float(f.subs(x, val)) for val in x_vals]

            iter_x = [row[1] for row in iteraciones]
            iter_y = [float(f.subs(x, val)) for val in iter_x]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, label=f'f(x) = {f_str}', color='blue')
            ax.plot(iter_x, iter_y, 'ro', label='Iteraciones')  # Graficar solo puntos
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la función y el proceso del Método de Newton')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['result'] = result
            context['iterations'] = iteraciones
            context['form'] = form
            context['image'] = uri

        else:
            context['form'] = form

        return render(request, self.template_name, context)

#---------SECANTE---------------------------------
class SecanteForm(forms.Form):
    x0 = forms.FloatField(label='X0', required=True)
    x1 = forms.FloatField(label='X1', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    n = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class SecantePageView(TemplateView):
    template_name = 'secante.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SecanteForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SecanteForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            try:
                x0 = float(form.cleaned_data['x0'])
                x1 = float(form.cleaned_data['x1'])
                tol = float(form.cleaned_data['tol'])
                n = int(form.cleaned_data['n'])
                fun_str = form.cleaned_data['fun']
                export = form.cleaned_data['export']
            except ValueError:
                context['error'] = "Asegúrate de que los valores numéricos están bien formateados."
                context['form'] = form
                return render(request, self.template_name, context)

            x = sp.symbols('x')
            try:
                f = sp.lambdify(x, sp.sympify(fun_str), 'numpy')
            except sp.SympifyError:
                context['error'] = "Error al parsear las expresiones. Asegúrate de usar la sintaxis correcta."
                context['form'] = form
                return render(request, self.template_name, context)

            def secante(x0, x1, tol, n, f):
                fx0 = f(x0)
                if fx0 == 0:
                    return x0, []
                else:
                    fx1 = f(x1)
                    i = 0
                    error = tol + 1
                    den = fx1 - fx0
                    table = []
                    while error > tol and fx1 != 0 and den != 0 and i < n:
                        x2 = x1 - ((fx1 * (x1 - x0)) / den)
                        error = np.abs(x2 - x1)
                        table.append((i, x0, x1, x2, fx1, error))
                        x0 = x1
                        fx0 = fx1
                        x1 = x2
                        fx1 = f(x1)
                        den = fx1 - fx0
                        i += 1
                    table.append((i, x0, x1, x2, fx1, error))
                    if fx1 == 0:
                        return [x1, "Error de " + str(0)], table
                    elif error < tol:
                        return [x1, "Error de " + str(error)], table
                    elif den == 0:
                        return ["Posible raíz múltiple"], table
                    else:
                        return [x1, "Fracasó en " + str(n) + " iteraciones"], table

            result, table = secante(x0, x1, tol, n, f)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_secante.txt"'
                response.write(f"Raíz aproximada: {result[0]}\n")
                response.write(f"{result[1]}\n")
                response.write("Iteración\tX0\tX1\tX2\tf(X1)\tError\n")
                for row in table:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            # Graficar la función y las iteraciones
            x_vals = np.linspace(min(x0, x1) - 2, max(x0, x1) + 2, 400)
            y_vals = [f(val) for val in x_vals]

            iter_x = [row[2] for row in table]
            iter_y = [f(val) for val in iter_x]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, label=f'f(x) = {fun_str}', color='blue')
            ax.plot(iter_x, iter_y, 'ro', label='Iteraciones')  # Graficar solo puntos
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la función y el proceso del Método de la Secante')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['result'] = result
            context['table'] = table
            context['form'] = form
            context['image'] = uri
        else:
            context['form'] = form

        return render(request, self.template_name, context)

#------------Raices_Multiples-----------------------------

class RaicesMultiplesForm(forms.Form):
    x0 = forms.FloatField(label='X0', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    n = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    deriv1 = forms.CharField(label='First Derivative', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la primera derivada en términos de x'}))
    deriv2 = forms.CharField(label='Second Derivative', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la segunda derivada en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class RaicesMultiplesPageView(TemplateView):
    template_name = 'raicesmultiples.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RaicesMultiplesForm()
        return context

    def post(self, request, *args, **kwargs):
        form = RaicesMultiplesForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            try:
                x0 = float(form.cleaned_data['x0'])
                tol = float(form.cleaned_data['tol'])
                n = int(form.cleaned_data['n'])
                fun_str = form.cleaned_data['fun']
                deriv1_str = form.cleaned_data['deriv1']
                deriv2_str = form.cleaned_data['deriv2']
                export = form.cleaned_data['export']
            except ValueError:
                context['error'] = "Asegúrate de que los valores numéricos están bien formateados."
                context['form'] = form
                return render(request, self.template_name, context)

            x = sp.symbols('x')
            try:
                f = sp.sympify(fun_str.replace('^', '**'))
                deriv1 = sp.sympify(deriv1_str.replace('^', '**'))
                deriv2 = sp.sympify(deriv2_str.replace('^', '**'))
            except sp.SympifyError:
                context['error'] = "Error al parsear las expresiones. Asegúrate de usar la sintaxis correcta."
                context['form'] = form
                return render(request, self.template_name, context)

            def RaicesMul(x0, tol, n, f, deriv1, deriv2):
                fx0 = f.subs(x, x0)
                table = []
                if fx0 == 0:
                    return x0, table
                else:
                    fpx0 = deriv1.subs(x, x0)
                    fppx0 = deriv2.subs(x, x0)
                    iter = 0
                    error = tol + 1
                    den = (fpx0**2) - (fx0 * fppx0)
                    while error > tol and fx0 != 0 and den != 0 and iter < n:
                        x1 = x0 - (fx0 * fpx0) / den
                        fx0 = f.subs(x, x1)
                        fpx0 = deriv1.subs(x, x1)
                        fppx0 = deriv2.subs(x, x1)
                        den = (fpx0**2) - (fx0 * fppx0)
                        error = abs(x1 - x0)
                        table.append((iter, x0, x1, fx0, error))
                        x0 = x1
                        iter += 1
                    table.append((iter, x0, x1, fx0, error))
                    if fx0 == 0:
                        return x0, table
                    elif error < tol:
                        return x0, table
                    else:
                        return x0, table

            result, table = RaicesMul(x0, tol, n, f, deriv1, deriv2)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_raicesmultiples.txt"'
                response.write(f"Raíz aproximada: {result}\n")
                response.write("Iteración\tX0\tX1\tf(X1)\tError\n")
                for row in table:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            # Graficar la función y las iteraciones
            x_vals = np.linspace(x0 - 2, x0 + 2, 400)
            y_vals = [f.subs(x, val) for val in x_vals]

            iter_x = [row[2] for row in table]
            iter_y = [f.subs(x, val) for val in iter_x]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, label=f'f(x) = {fun_str}', color='blue')
            ax.plot(iter_x, iter_y, 'ro', label='Iteraciones')  # Graficar solo puntos
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la función y el proceso del Método de Raíces Múltiples')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['result'] = result
            context['table'] = table
            context['form'] = form
            context['image'] = uri
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#------------------GaussSeidel----------------------------- 
class GaussSeidelForm(forms.Form):
    rows = forms.IntegerField(label="Número de filas")
    cols = forms.IntegerField(label="Número de columnas")
    tol = forms.FloatField(label="Tolerancia")
    niter = forms.IntegerField(label="Número máximo de iteraciones")
    export = forms.BooleanField(required=False, label="Exportar resultados a TXT")
    
class GaussSeidelPageView(TemplateView):
    template_name = 'gaussseidel.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = GaussSeidelForm()
        return context

    def post(self, request, *args, **kwargs):
        form = GaussSeidelForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['rows']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            try:
                for i in range(n):
                    for j in range(n):
                        A[i][j] = float(request.POST.get(f'A_{i}_{j}', 0))
                    b[i] = float(request.POST.get(f'b_{i}', 0))
                    x0[i] = float(request.POST.get(f'x0_{i}', 0))
            except ValueError as e:
                context['error'] = "Asegúrate de ingresar todos los valores de la matriz y el vector correctamente."
                context['form'] = form
                return render(request, self.template_name, context)

            def gauss_seidel(A, b, x0, n, tol, niter):
                # Check for zero in the diagonal
                if any(A[i][i] == 0 for i in range(n)):
                    return None, "El método no puede converger debido a un 0 en la diagonal de la matriz A."

                iteraciones = []
                it = 0
                t = tol + 1
                iteraciones.append([it, x0.copy().tolist(), t])
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.zeros(n)
                    for j in range(n):
                        sum_ax = sum(A[j][k] * x_nuevo[k] if k < j else A[j][k] * x0[k] for k in range(n) if k != j)
                        x_nuevo[j] = (b[j] - sum_ax) / A[j][j]
                    t = np.max(np.abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    iteraciones.append([it, x0.copy().tolist(), t])
                return iteraciones, x0.tolist()

            iterations, result = gauss_seidel(A, b, x0, n, tol, niter)

            if iterations is None:
                context['error'] = result
                context['form'] = form
                return render(request, self.template_name, context)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_gaussseidel.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iterations:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)

    
#---------------------Jacobi------------------------------------------------
class JacobiForm(forms.Form):
    n = forms.IntegerField(label='Dimensión de la matriz (n)', required=True)
    tol = forms.FloatField(label='Tolerancia', required=True)
    niter = forms.IntegerField(label='Número máximo de iteraciones', required=True)
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class JacobiPageView(TemplateView):
    template_name = 'jacobi.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = JacobiForm()
        return context

    def post(self, request, *args, **kwargs):
        form = JacobiForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    value = request.POST.get(f'A_{i}_{j}')
                    if value is not None and value != '':
                        A[i][j] = float(value)
                    else:
                        context['error'] = "Todos los valores de la matriz A deben ser proporcionados."
                        return render(request, self.template_name, context)

                value = request.POST.get(f'b_{i}')
                if value is not None and value != '':
                    b[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector b deben ser proporcionados."
                    return render(request, self.template_name, context)

                value = request.POST.get(f'x0_{i}')
                if value is not None and value != '':
                    x0[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector x0 deben ser proporcionados."
                    return render(request, self.template_name, context)

            # Verificar ceros en la diagonal de la matriz
            if any(A[i][i] == 0 for i in range(n)):
                context['error'] = "El método no puede converger debido a un 0 en la diagonal de la matriz A."
                return render(request, self.template_name, context)

            def jacobi(A, b, x0, n, tol, niter):
                iteraciones = []
                it = 0
                t = tol + 1
                iteraciones.append([it, x0.copy().tolist(), t])
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.zeros(n)
                    for j in range(n):
                        suma = sum(-A[j][k] * x0[k] if j != k else 0 for k in range(n))
                        x_nuevo[j] = (b[j] + suma) / A[j][j]
                    t = np.max(np.abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    iteraciones.append([it, x0.copy().tolist(), t])
                return iteraciones, x0.tolist()

            iteraciones, result = jacobi(A, b, x0, n, tol, niter)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_jacobi.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iteraciones:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iteraciones
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)


    
#-------------SOR--------------------------------------------------------
class SORForm(forms.Form):
    n = forms.IntegerField(label="Dimensiones de la matriz (n)")
    tol = forms.FloatField(label="Tolerancia")
    niter = forms.IntegerField(label="Número máximo de iteraciones")
    w = forms.FloatField(label="Parámetro de relajación (ω)")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)
    

class SORPageView(TemplateView):
    template_name = 'sor.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SORForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SORForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            w = form.cleaned_data['w']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    value = request.POST.get(f'A_{i}_{j}')
                    if value is not None and value != '':
                        A[i][j] = float(value)
                    else:
                        context['error'] = "Todos los valores de la matriz A deben ser proporcionados."
                        return render(request, self.template_name, context)

                value = request.POST.get(f'b_{i}')
                if value is not None and value != '':
                    b[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector b deben ser proporcionados."
                    return render(request, self.template_name, context)

                value = request.POST.get(f'x0_{i}')
                if value is not None and value != '':
                    x0[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector x0 deben ser proporcionados."
                    return render(request, self.template_name, context)

            # Verificar ceros en la diagonal de la matriz
            if any(A[i][i] == 0 for i in range(n)):
                context['error'] = "El método no puede converger debido a un 0 en la diagonal de la matriz A."
                return render(request, self.template_name, context)

            def sor(A, b, x0, n, tol, w):
                iteraciones = []
                it = 0
                t = tol + 1
                iteraciones.append([it, x0.copy().tolist(), t])
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.zeros(n)
                    for j in range(n):
                        sum_ax = sum(A[j][k] * x_nuevo[k] if k < j else A[j][k] * x0[k] for k in range(n) if k != j)
                        x_nuevo[j] = (b[j] - sum_ax) / A[j][j]
                        x_nuevo[j] = (1 - w) * x0[j] + w * x_nuevo[j]
                    t = np.max(np.abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    iteraciones.append([it, x0.copy().tolist(), t])
                return iteraciones, x0.tolist()

            iterations, result = sor(A, b, x0, n, tol, w)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_sor.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iterations:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)



#------------------SORMatricial-----------------------------------  
class SORMForm(forms.Form):
    n = forms.IntegerField(label="Dimensiones de la matriz (n)")
    tol = forms.FloatField(label="Tolerancia")
    niter = forms.IntegerField(label="Número máximo de iteraciones")
    w = forms.FloatField(label="Parámetro de relajación (ω)")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False) 
    
class SORMatricialPageView(TemplateView):
    template_name = 'sor_matricial.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SORMForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SORMForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            w = form.cleaned_data['w']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    A[i][j] = float(request.POST.get(f'A_{i}_{j}'))
                b[i] = float(request.POST.get(f'b_{i}'))
                x0[i] = float(request.POST.get(f'x0_{i}'))

            def sor_matricial(A, b, x0, n, tol, w):
                table = []
                it = 0
                t = tol + 1
                table.append([it, x0.copy().tolist(), t])
                D = np.diag(np.diag(A))
                L = (-1) * np.tril(A - D)
                U = (-1) * np.triu(A - D)
                T = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)
                C = w * np.linalg.inv(D - w * L) @ np.transpose(b)
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.transpose(T @ np.transpose(x0) + C)
                    t = max(abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    table.append([it, x0.copy().tolist(), t])
                return table, x0.tolist()

            iterations, result = sor_matricial(A, b, x0, n, tol, w)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_sor_matricial.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iterations:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)

#-------------------Vandermonde----------------------------

class VandermondeForm(forms.Form):
    x_vals = forms.CharField(label='X values', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese los valores de x separados por comas'}))
    y_vals = forms.CharField(label='Y values', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese los valores de y separados por comas'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class VandermondePageView(TemplateView):
    template_name = 'vandermonde.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = VandermondeForm()
        return context

    def post(self, request, *args, **kwargs):
        form = VandermondeForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            try:
                x_values = list(map(float, form.cleaned_data['x_vals'].split(',')))
                y_values = list(map(float, form.cleaned_data['y_vals'].split(',')))
                export = form.cleaned_data['export']
            except ValueError:
                context['error'] = "Asegúrate de que los valores de x e y están bien formateados y separados por comas."
                context['form'] = form
                return render(request, self.template_name, context)

            if len(x_values) != len(y_values):
                context['error'] = "El número de valores de x y y deben ser iguales."
                context['form'] = form
                return render(request, self.template_name, context)

            n = len(x_values)
            A = np.vander(x_values, increasing=True)
            coef = np.linalg.solve(A, y_values)

            # Crear la función de interpolación
            x = sp.symbols('x')
            poly = sum(sp.Rational(c) * x**i for i, c in enumerate(coef))

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_vandermonde.txt"'
                response.write("Coeficientes del polinomio de interpolación:\n")
                response.write(f"{coef}\n")
                return response

            # Graficar los puntos y el polinomio de interpolación
            x_plot = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
            y_plot = [poly.subs(x, val) for val in x_plot]
            y_plot = np.array(y_plot, dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y_plot, label=f'Polinomio de interpolación', color='blue')
            ax.plot(x_values, y_values, 'ro', label='Puntos de interpolación')  # Graficar solo puntos
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la interpolación de Vandermonde')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['coeffs'] = coef
            context['form'] = form
            context['image'] = uri
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
    
#---------Newton_Interpolante--------------------------

class NewtonInterpolanteForm(forms.Form):
    n = forms.IntegerField(label="Número de puntos")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)

class NewtonInterpolantePageView(TemplateView):
    template_name = 'newton_interpolante.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = NewtonInterpolanteForm()
        return context

    def post(self, request, *args, **kwargs):
        form = NewtonInterpolanteForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            export = form.cleaned_data['export']
            try:
                xs = [float(request.POST.get(f'xs_{i}')) for i in range(n)]
                y = [float(request.POST.get(f'y_{i}')) for i in range(n)]
            except (TypeError, ValueError) as e:
                context['error'] = "Asegúrate de que todos los valores de xs y y están correctamente ingresados y son números válidos."
                context['form'] = form
                return render(request, self.template_name, context)

            def newton_interpolante(xs, y, n):
                x = sp.symbols("x")
                dif_div = np.zeros((n, n + 1))
                dif_div[:, 0] = xs
                dif_div[:, 1] = y
                expr = 0
                for i in range(n - 1):
                    for j in range(n - 1):
                        if i <= j:
                            dif_div[j + 1][i + 2] = (dif_div[j + 1][i + 1] - dif_div[j][i + 1]) / (dif_div[j + 1][0] - dif_div[j + 1 - i - 1][0])
                for i in range(n):
                    aux = dif_div[i][i + 1]
                    for j in range(i):
                        aux *= (x - dif_div[j][0])
                    expr += aux
                expr = sp.expand(expr)
                funcion = f"F(x) = {expr}"
                return funcion, expr

            funcion, expr = newton_interpolante(xs, y, n)
            context['funcion'] = funcion
            context['expr'] = expr
            context['form'] = form

            # Graficar los puntos y el polinomio de interpolación
            x_plot = np.linspace(min(xs) - 1, max(xs) + 1, 400)
            y_plot = [expr.subs(sp.symbols('x'), val) for val in x_plot]
            y_plot = np.array(y_plot, dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y_plot, label='Polinomio de interpolación', color='blue')
            ax.plot(xs, y, 'ro', label='Puntos de interpolación')  # Graficar solo puntos
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la interpolación de Newton')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['image'] = uri

            # Lógica de exportación
            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_newton_interpolante.txt"'
                response.write("Coeficientes del polinomio de interpolación de Newton:\n")
                response.write(f"{expr}\n")
                response.write("Puntos de interpolación:\n")
                for i in range(n):
                    response.write(f"x_{i}: {xs[i]}, y_{i}: {y[i]}\n")
                return response

        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#--------lagrange---------------------------------------------------

class LagrangeForm(forms.Form):
    n = forms.IntegerField(label="Número de puntos")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)
    
class LagrangePageView(TemplateView):
    template_name = 'lagrange.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = LagrangeForm()
        return context

    def post(self, request, *args, **kwargs):
        form = LagrangeForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            export = form.cleaned_data['export']
            try:
                xs = [float(request.POST.get(f'xs_{i}')) for i in range(n)]
                y = [float(request.POST.get(f'y_{i}')) for i in range(n)]
            except (TypeError, ValueError) as e:
                context['error'] = "Asegúrate de que todos los valores de xs y y están correctamente ingresados y son números válidos."
                context['form'] = form
                return render(request, self.template_name, context)

            def lagrange(xs, y, n):
                x = sp.symbols('x')
                expr = 0
                for i in range(n):
                    aux = 1
                    for j in range(n):
                        if i != j:
                            aux *= (x - xs[j]) / (xs[i] - xs[j])
                    expr += aux * y[i]
                expr = sp.expand(expr)
                funcion = f"F(x) = {expr}"
                return funcion, expr

            funcion, expr = lagrange(xs, y, n)
            context['funcion'] = funcion
            context['expr'] = expr
            context['form'] = form

            # Graficar los puntos y el polinomio de interpolación
            x_plot = np.linspace(min(xs) - 1, max(xs) + 1, 400)
            y_plot = [expr.subs(sp.symbols('x'), val) for val in x_plot]
            y_plot = np.array(y_plot, dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y_plot, label='Polinomio de interpolación', color='blue')
            ax.plot(xs, y, 'ro', label='Puntos de interpolación')  # Graficar solo puntos
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la interpolación de Lagrange')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['image'] = uri

            # Lógica de exportación
            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_lagrange.txt"'
                response.write("Polinomio Interpolante de Lagrange:\n")
                response.write(f"{expr}\n")
                response.write("Puntos de interpolación:\n")
                for i in range(n):
                    response.write(f"x_{i}: {xs[i]}, y_{i}: {y[i]}\n")
                return response

        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#----------------------spline-----------------------------------

class SplineForm(forms.Form):
    n = forms.IntegerField(label='Número de puntos', min_value=2, required=True)
    xs = forms.CharField(label='Valores de X (separados por comas)', required=True)
    y = forms.CharField(label='Valores de Y (separados por comas)', required=True)
    export_txt = forms.BooleanField(label='Exportar resultados a TXT', required=False)
    

class SplineLinealPageView(TemplateView):
    template_name = 'spline.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SplineForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SplineForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            xs = np.array([float(x.strip()) for x in form.cleaned_data['xs'].split(',')])
            y = np.array([float(y.strip()) for y in form.cleaned_data['y'].split(',')])

            coef, polinomios = self.spline_lineal(xs, y, n)

            context['coeficientes'] = coef.tolist()
            context['polinomios'] = polinomios
            context['form'] = form

            # Graficar los puntos y el polinomio de interpolación
            x_plot = np.linspace(min(xs), max(xs), 400)
            y_plot = np.zeros_like(x_plot)
            for i in range(n - 1):
                indices = (x_plot >= xs[i]) & (x_plot <= xs[i + 1])
                y_plot[indices] = coef[i, 0] * x_plot[indices] + coef[i, 1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y_plot, label='Polinomio de interpolación', color='blue')
            ax.plot(xs, y, 'ro', label='Puntos de interpolación')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfica de la interpolación de Spline Lineal')
            plt.legend()

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['image'] = uri

            if form.cleaned_data.get('export_txt', False):
                response = self.export_to_txt(coef, polinomios)
                return response

        context['form'] = form
        return render(request, self.template_name, context)

    def spline_lineal(self, xs, y, n):
        coef = np.zeros((n-1, 2))
        polinomios = []
        for i in range(n-1):
            A = np.array([[xs[i], 1], [xs[i+1], 1]])
            B = np.array([[y[i]], [y[i+1]]])
            coef[i, :] = np.transpose(np.linalg.inv(A) @ B)
            polinomios.append(f"P{i+1}(x) = {round(coef[i][0], 4)}x + {round(coef[i][1], 4)}")
        return coef, polinomios

    def export_to_txt(self, coef, polinomios):
        content = 'Coeficientes:\n'
        for row in coef:
            content += f'{row[0]} {row[1]}\n'
        content += '\nPolinomios:\n'
        for poly in polinomios:
            content += f'{poly}\n'
        
        response = HttpResponse(content, content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="resultados.txt"'
        return response

class SplineCubicoForm(forms.Form):
    x_values = forms.CharField(label='Valores de X (separados por comas)', required=True, 
                             widget=forms.TextInput(attrs={'placeholder': 'Ejemplo: 1,2,3,4,5'}))
    y_values = forms.CharField(label='Valores de Y (separados por comas)', required=True,
                             widget=forms.TextInput(attrs={'placeholder': 'Ejemplo: 2,4,6,8,10'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class SplineCubicoPageView(TemplateView):
    template_name = 'spline_cubico.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SplineCubicoForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SplineCubicoForm(request.POST)
        context = self.get_context_data()
        
        if form.is_valid():
            try:
                x_values = [float(x.strip()) for x in form.cleaned_data['x_values'].split(',')]
                y_values = [float(y.strip()) for y in form.cleaned_data['y_values'].split(',')]
                
                if len(x_values) != len(y_values):
                    context['error'] = 'Los vectores X e Y deben tener la misma longitud'
                    context['form'] = form
                    return render(request, self.template_name, context)
                
                if len(x_values) < 4:
                    context['error'] = 'Se requieren al menos 4 puntos para el Spline Cúbico'
                    context['form'] = form
                    return render(request, self.template_name, context)

                funcion, expr, image = interpolacion_spline_cubico(x_values, y_values)
                
                context['funcion'] = funcion
                context['expr'] = expr
                context['image'] = image
                context['form'] = form

            except ValueError as e:
                context['error'] = str(e)
                context['form'] = form
            except Exception as e:
                context['error'] = f'Error al procesar los datos: {str(e)}'
                context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)

def home(request):
    return render(request, 'home.html')

def metodos(request):
    return render(request, 'metodos.html')

def nosotros(request):
    return render(request, 'nosotros.html')

def ayuda_derivadas(request):
    return render(request, 'ayuda/derivadas.html')

@csrf_exempt
def interpolacion(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            x_values = [float(x) for x in data['x_values'].split(',')]
            y_values = [float(y) for y in data['y_values'].split(',')]
            metodo = data['metodo']
            
            # Convertir a arrays de numpy
            x = np.array(x_values)
            y = np.array(y_values)
            
            if metodo == 'vandermonde':
                polinomio, error, graph = interpolacion_vandermonde(x_values, y_values)
                
            elif metodo == 'newton':
                polinomio, error, graph = newtonint(x_values, y_values)
                
            elif metodo == 'lagrange':
                polinomio, error, graph = lagrange(x_values, y_values)
                
            elif metodo == 'spline_lineal':
                polinomio, error, graph = interpolacion_spline_lineal(x_values, y_values)
                
            elif metodo == 'spline_cubico':
                polinomio, error, graph = interpolacion_spline_cubico(x_values, y_values)
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Método no válido'
                })
            
            return JsonResponse({
                'success': True,
                'polinomio': polinomio,
                'error': error,
                'graph': graph
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Método no permitido'
    })

def vandermonde(request):
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            x_values = [float(x) for x in request.POST.get('x_values', '').split(',')]
            y_values = [float(y) for y in request.POST.get('y_values', '').split(',')]
            
            polinomio, error, graph = interpolacion_vandermonde(x_values, y_values)
            
            return JsonResponse({
                'success': True,
                'graph': graph,
                'polinomio': polinomio,
                'error': error
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return render(request, 'vandermonde.html')

class ComparacionGeneralPageView(TemplateView):
    template_name = 'comparacion_general.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def post(self, request, *args, **kwargs):
        # Recopilar resultados de todos los métodos
        resultados = {}
        
        # Capítulo 1: Métodos de Búsqueda de Raíces
        for nombre, metodo in ComparadorGeneral().metodos_cap1.items():
            try:
                # Usar parámetros por defecto para la comparación
                iterations, table, graph, report = metodo.solve(
                    a=-1, b=1, tol=1e-6, niter=100,
                    fun="x^2 - 4"  # Función de ejemplo
                )
                resultados[nombre] = {
                    'capitulo': 'cap1',
                    'iteraciones': len(iterations),
                    'error': iterations[-1][3] if iterations else float('inf'),
                    'converge': True
                }
            except Exception as e:
                resultados[nombre] = {
                    'capitulo': 'cap1',
                    'iteraciones': 0,
                    'error': float('inf'),
                    'converge': False,
                    'error_msg': str(e)
                }
        
        # Capítulo 2: Métodos de Sistemas Lineales
        for nombre, metodo in ComparadorGeneral().metodos_cap2.items():
            try:
                # Usar matriz de ejemplo 3x3
                A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
                b = np.array([1, 1, 1])
                x0 = np.zeros(3)
                iterations, result = metodo.solve(A, b, x0, 3, 1e-6, 100)
                resultados[nombre] = {
                    'capitulo': 'cap2',
                    'iteraciones': len(iterations),
                    'error': iterations[-1][2] if iterations else float('inf'),
                    'converge': True
                }
            except Exception as e:
                resultados[nombre] = {
                    'capitulo': 'cap2',
                    'iteraciones': 0,
                    'error': float('inf'),
                    'converge': False,
                    'error_msg': str(e)
                }
        
        # Capítulo 3: Métodos de Interpolación
        x_values = [0, 1, 2, 3, 4]
        y_values = [0, 1, 4, 9, 16]
        for nombre, metodo in ComparadorGeneral().metodos_cap3.items():
            try:
                if nombre in ['Vandermonde', 'Newton Interpolante', 'Lagrange']:
                    polinomio, error, graph = metodo(x_values, y_values)
                    resultados[nombre] = {
                        'capitulo': 'cap3',
                        'iteraciones': 1,  # Los métodos de interpolación no son iterativos
                        'error': error,
                        'converge': True
                    }
                else:  # Spline Lineal y Cúbico
                    coef, polinomios = metodo(x_values, y_values, len(x_values))
                    resultados[nombre] = {
                        'capitulo': 'cap3',
                        'iteraciones': 1,
                        'error': 0,  # Los splines pasan exactamente por los puntos
                        'converge': True
                    }
            except Exception as e:
                resultados[nombre] = {
                    'capitulo': 'cap3',
                    'iteraciones': 0,
                    'error': float('inf'),
                    'converge': False,
                    'error_msg': str(e)
                }
        
        # Generar informe de comparación
        comparador = ComparadorGeneral()
        informe = comparador.comparar_todos(resultados)
        
        context = self.get_context_data()
        context['informe'] = informe
        return render(request, self.template_name, context)