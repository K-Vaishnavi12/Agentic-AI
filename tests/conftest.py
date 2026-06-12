"""Pytest configuration & shared fixtures."""
import os
import sys
from pathlib import Path

# Make ``src`` importable when running pytest from the project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use ephemeral data dir for tests so the user's models aren't touched
os.environ.setdefault("ML_SKIP_TRANSFORMER", "1")
