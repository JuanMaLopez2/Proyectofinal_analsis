# Este archivo permite que Python reconozca el directorio como un subpaquete 

from .newtonint import newtonint
from .newtonor import newtonor
from .lagrange import lagrange
from .spline_lineal import interpolacion_spline_lineal
from .spline_cubico import interpolacion_spline_cubico
from .vandermonde import interpolacion_vandermonde

__all__ = [
    'newtonint',
    'newtonor',
    'lagrange',
    'interpolacion_spline_lineal',
    'interpolacion_spline_cubico',
    'interpolacion_vandermonde'
] 