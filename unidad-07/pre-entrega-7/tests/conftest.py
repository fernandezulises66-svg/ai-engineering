import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PHOENIX_ENABLED"] = "false"      # los tests no mandan trazas
os.environ["RESEARCH_BACKEND"] = "local"
