"""
Métodos del Capítulo 1: Búsqueda de raíces.
"""

from .base import RootFindingMethod
from .biseccion import Biseccion
from .ReglaFalsa import ReglaFalsa
from .Puntofijo import PuntoFijo
from .Newton import Newton
from .Secante import Secante
from .RaicesMultiples import RaicesMultiples

__all__ = [
    'RootFindingMethod',
    'Biseccion',
    'ReglaFalsa',
    'PuntoFijo',
    'Newton',
    'Secante',
    'RaicesMultiples'
] 