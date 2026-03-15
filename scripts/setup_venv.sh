#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3.10}
VENV_DIR=${VENV_DIR:-.venv}
INSTALL_TORCH_CPU=${INSTALL_TORCH_CPU:-1}

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements-py310.txt
if [ "$INSTALL_TORCH_CPU" = "1" ]; then
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu
fi

# Ray 2.x compatibility shim for newer setuptools/pkg_resources behavior.
python - <<'PY'
import site
from pathlib import Path

site_pkgs = next(p for p in site.getsitepackages() if "site-packages" in p)
shim = Path(site_pkgs) / "sitecustomize.py"
shim.write_text(
    "import pkg_resources\n"
    "import packaging\n"
    "pkg_resources.packaging = packaging\n",
    encoding="utf-8",
)
print(f"installed: {shim}")
PY

echo "venv ready: $VENV_DIR"
