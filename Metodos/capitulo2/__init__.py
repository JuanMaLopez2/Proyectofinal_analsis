"""
Métodos del Capítulo 2: Sistemas lineales.
"""

from .jacobi import Jacobi
from .gaussSeidel import GaussSeidel
from .sor import SOR
from .sorMatricial import SORMatricial

__all__ = [
    'Jacobi',
    'GaussSeidel',
    'SOR',
    'SORMatricial'
] 