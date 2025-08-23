import sys
from pathlib import Path

# so `import src.*` / `import shared.*` work no matter where pytest is invoked
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
