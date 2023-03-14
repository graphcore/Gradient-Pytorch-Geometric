

import sys
import pathlib
_nbfnet_code_directory = pathlib.Path(__file__).resolve().parent / "nbfnet_utils"
sys.path.insert(0, str(_nbfnet_code_directory))

# Make it look like this file is the original notebook_utils
from nbfnet_utils import *