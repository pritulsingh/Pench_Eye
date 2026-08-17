"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

# Allow `import app.*` and `import ml.*` when pytest is invoked from anywhere.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
for path in (BACKEND_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))