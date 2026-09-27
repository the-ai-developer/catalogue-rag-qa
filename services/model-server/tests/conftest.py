"""Make `pytest` work from any cwd: put the model-server package on sys.path."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
