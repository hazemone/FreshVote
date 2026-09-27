"""
Adds code/ to sys.path so tests can `from config import ...` etc.
Run from code/ with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
