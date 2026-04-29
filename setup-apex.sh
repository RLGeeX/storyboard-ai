#!/usr/bin/env bash
# Setup script for storyboard-ai using mise (Python toolchain) and the
# existing Vertex AI configuration from cm-vertex-demo.
#
# Run from the storyboard-ai/ directory:   bash setup-apex.sh
#
# What it does:
#   1. Verifies mise is installed
#   2. Runs `mise install` to provision Python 3.12 (per .mise.toml) + auto-create .venv
#   3. Runs the `setup` task in .mise.toml (pip install of all deps)
#   4. Verifies ffmpeg is present
#   5. Confirms the existing Vertex AI service-account key is reachable
#   6. Seeds genai-pipeline/.env from the template if missing

set -e

echo "=== storyboard-ai setup (mise + Vertex AI) ==="

# ---- mise check ----
if ! command -v mise >/dev/null 2>&1; then
  echo "ERROR: mise not found on PATH. Install with:  brew install mise"
  echo "Then re-run this script."
  exit 1
fi
echo "Found mise: $(mise --version)"

# Ensure mise trusts this directory's .mise.toml
mise trust . >/dev/null 2>&1 || true

# ---- ffmpeg check ----
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install with:  brew install ffmpeg"
  exit 1
fi
echo "Found ffmpeg: $(ffmpeg -version | head -1)"

# ---- Python install via mise ----
echo "Provisioning Python via mise (this also creates .venv on first run)..."
mise install
PY_VERSION=$(mise exec -- python --version 2>&1)
echo "mise Python: $PY_VERSION"

# ---- Install Python deps via the mise task ----
echo "Installing Python dependencies (mise run setup)..."
mise run setup

# ---- Verify the existing GCP service-account key is reachable ----
SA_KEY="/Users/jfogarty/prj/chain-mountain/cm-iq/cm-iq-app/secrets/cm-iq-sa-key.json"
if [ -f "$SA_KEY" ]; then
  echo "Found service-account key at $SA_KEY"
else
  echo "WARNING: SA key not found at $SA_KEY"
  echo "         If you've moved it, edit GOOGLE_APPLICATION_CREDENTIALS in genai-pipeline/.env"
fi

# ---- Seed .env from template ----
if [ ! -f "genai-pipeline/.env" ]; then
  if [ -f "genai-pipeline/.env.template" ]; then
    cp genai-pipeline/.env.template genai-pipeline/.env
    echo "Created genai-pipeline/.env from template (pre-filled with cm-iq-demo / us-east1)."
  fi
else
  echo "genai-pipeline/.env already exists — leaving it alone."
fi

# ---- Optional preflight ----
echo ""
echo "Running preflight checks..."
mise run preflight || {
  echo ""
  echo "Preflight reported issues. Common fixes:"
  echo "  - 'genai client OK' fails -> check Vertex env vars in genai-pipeline/.env"
  echo "  - 'imports OK' fails       -> rerun  mise run setup"
}

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. cd into the storyboard-ai folder so mise activates the venv:  cd ."
echo "  2. (Optional) Verify .env is correct:  cat genai-pipeline/.env"
echo "  3. Run the pipeline:  mise run run"
echo ""
echo "See ../Chain Mountain/apex-storyboard-ai-setup.md for the full walkthrough."
