import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from . import utils, filters, signal_generators, synth_utils, envelope_adsr, visualization
from .utils import *
from .filters import *
from .signal_generators import *
from .synth_utils import *
from .envelope_adsr import *
from .visualization import *

globals().update({
    "np": np,
    "plt": plt,
    "sys": sys,
    "Path": Path
})
