#!/usr/bin/env bash
# Measure every label on every figure against the width the manuscript places it
# at, and refuse to stay quiet about one that is too small or that sits on top of
# another.
#
#   ./checkfigs.sh                       every figure in figures/
#   ./checkfigs.sh figures/fig_map.pdf   one of them
#   ./checkfigs.sh --width 19.0          a different placed width
#
# figcheck needs pdfplumber, and pdfplumber needs a Pillow newer than veros,
# moviepy and streamlit accept, so installing it in front of the conda
# environment breaks those. It belongs in an environment of its own:
#
#     python3 -m venv .venv-figcheck
#     .venv-figcheck/bin/pip install -q pdfplumber
#
# This prefers that environment, then the identical one in the companion plume
# paper, and only then the interpreter on the path.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
for c in .venv-figcheck/bin/python3 \
         ../Paper_plumes/plume_pipeline/.venv-figcheck/bin/python3; do
    if [ -x "$c" ] && "$c" -c 'import pdfplumber' 2>/dev/null; then PY="$c"; break; fi
done

if ! "$PY" -c 'import pdfplumber' 2>/dev/null; then
    echo "figcheck needs pdfplumber and no environment here has it." >&2
    echo "Do not install it into the conda environment; it will pull a Pillow" >&2
    echo "that veros, moviepy and streamlit do not accept. Make it its own:" >&2
    echo >&2
    echo "    python3 -m venv .venv-figcheck" >&2
    echo "    .venv-figcheck/bin/pip install -q pdfplumber" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    exec "$PY" src/figcheck.py "$@"
fi
if ! compgen -G 'figures/*.pdf' > /dev/null; then
    echo "no figures in figures/ yet; run the fig_*.py scripts first (see RUN.md)" >&2
    exit 1
fi
exec "$PY" src/figcheck.py figures/*.pdf
