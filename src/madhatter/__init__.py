"""
    Main class
"""

from .typings.benchmark import *
from .typings.models import *
from .typings.utils import *

from .loaders import load_concreteness, load_freq, load_imageability
import os.path

if not os.path.exists(f"{__package__}/static"):
    load_freq()
    load_concreteness()
    load_imageability()
