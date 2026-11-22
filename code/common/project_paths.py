from pathlib import Path
import sys
import os

# this file lives in code/common/ ; code/ holds the experiment packages + vendored mycode/
COMMON_DIR = Path(__file__).resolve().parent
CODE_DIR = COMMON_DIR.parent           # code/
PROJECT_DIR = CODE_DIR.parent          # GRAIL/
ROOT_DIR = PROJECT_DIR.parent          # ZK/ (repo root, holds the shared data/)

# Result files and figures live at the artifact root
RESULTS_DIR = PROJECT_DIR / "results"
IMAGES_DIR = PROJECT_DIR / "figures"

# The dataset is shared at the repo root; the vendored mycode/args.py honours GRAIL_DATA_ROOT
os.environ.setdefault("GRAIL_DATA_ROOT", str(ROOT_DIR / "data" / "cyberdata"))

# Make the vendored mycode/ importable (code/mycode/)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
