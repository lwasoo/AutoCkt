#!/usr/bin/env bash
set -euo pipefail

# One-shot native setup for Linux (non-Docker):
# 1) install system deps (best effort)
# 2) create venv + install Python deps

SUDO=${SUDO:-sudo}

if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update
  $SUDO apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    ngspice \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils
elif command -v yum >/dev/null 2>&1; then
  $SUDO yum install -y \
    gcc \
    gcc-c++ \
    make \
    wget \
    curl \
    git \
    ngspice \
    libffi-devel \
    openssl-devel \
    zlib-devel \
    bzip2-devel \
    readline-devel \
    sqlite-devel \
    ncurses-devel \
    xz-devel
else
  echo "Unsupported package manager. Please install ngspice + build deps manually."
fi

bash scripts/setup_venv.sh

echo "Native setup complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python eval_engines/ngspice/ngspice_inputs/correct_inputs.py"

