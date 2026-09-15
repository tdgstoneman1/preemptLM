#!/usr/bin/env bash
# Creates 'converted-models/', 'expert-bank/', and 'traces/' local directories,
# sets up virtual environment and installs dependencies, and makes scripts executable.
# Requires uv

set -euo pipefail

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m==> %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m==> %s\033[0m\n' "$*" >&2; exit 1; }

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [ "${SOURCE#/}" = "$SOURCE" ] && SOURCE="$DIR/$SOURCE"
done

SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO="$(cd -P "$SCRIPT_DIR/.." && pwd)"

# Check prerequisites
command -v uv >/dev/null 2>&1 || die "uv not found. Install uv first: https://docs.astral.sh/uv/"

# Create directories
say "Creating required project directories..."
mkdir -p "$REPO/traces"
mkdir -p "$REPO/expert-bank"
mkdir -p "$REPO/converted-models"
say "Created directories: traces/, expert-bank/, converted-models/"

# Download python deps
say "Configuring virtual environment and syncing dependencies with uv..."
(cd "$REPO" && uv sync)
say "Virtual environment synced successfully."

# 4. Make scripts executable
say "Setting script permissions..."
chmod +x "$SCRIPT_DIR/run_host_bridge_mcp.sh" "$SCRIPT_DIR/setup.sh"
say "Setup complete."
