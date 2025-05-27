class PuntoFijoForm(forms.Form):
    x0 = forms.FloatField(
        label='Valor inicial (x0)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    tol = forms.FloatField(
        label='Tolerancia',
        initial=1e-6,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    niter = forms.IntegerField(
        label='Número máximo de iteraciones',
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    fun = forms.CharField(
        label='Función f(x)',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: x^2 - 4'})
    )
    g = forms.CharField(
        label='Función de iteración g(x)',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: sqrt(x + 4)'})
    )
    export = forms.BooleanField(
        label='Exportar resultados',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    ) 