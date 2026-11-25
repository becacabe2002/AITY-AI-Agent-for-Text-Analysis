"""The shim is not part of aity_web; put its folder on sys.path for its tests."""
import sys
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parents[2] / "docker" / "ollama" / "shim"
sys.path.insert(0, str(SHIM_DIR))
