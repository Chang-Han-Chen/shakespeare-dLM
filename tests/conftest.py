"""
conftest.py — Ensure the repo root is on sys.path so that
imports like `from backbone import DiffusionBackbone` resolve
when tests are run from anywhere via `pytest tests/`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
