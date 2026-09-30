import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parent.parent
REPO = PROBE.parent.parent

# The probe's own modules are imported as top-level names (embed, sweep, ...)
# rather than as a package called `src`, because benchmark/ already contributes
# a top-level `src` package and two of those cannot coexist on one path.
for p in (str(PROBE / "src"), str(PROBE), str(REPO / "benchmark"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
