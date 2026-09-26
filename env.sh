# Source from anywhere:  source env.sh
# Points the code at the bundled datasets and makes code/ and code/common/ importable.
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRAIL_DATA_ROOT="$ART/data/cyberdata"
export PYTHONPATH="$ART/code:$ART/code/common${PYTHONPATH:+:$PYTHONPATH}"
echo "GRAIL_DATA_ROOT=$GRAIL_DATA_ROOT"
