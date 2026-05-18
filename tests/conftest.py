import sys
from pathlib import Path

# Make `from src.xxx import yyy` resolve against benchmark/src/
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
