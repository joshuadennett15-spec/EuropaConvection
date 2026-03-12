"""
Europa 2D Axisymmetric Ice Shell Model.

Extends EuropaProjectDJ's 1D thermal solver to a coupled-column
2D axisymmetric model with latitude-dependent physics.
"""
import sys
import os

# Add EuropaProjectDJ/src to Python path so we can import its modules
_PROJ_1D = os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src')
if os.path.isdir(_PROJ_1D) and _PROJ_1D not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJ_1D))
