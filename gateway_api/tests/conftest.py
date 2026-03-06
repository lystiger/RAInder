import sys
from pathlib import Path

# Ensure `app` package imports resolve when tests run from repo root or gateway_api dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
