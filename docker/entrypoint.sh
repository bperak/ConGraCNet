#!/usr/bin/env bash
set -euo pipefail

# Create runtime config directory that will be prepended to PYTHONPATH
RUNTIME_DIR="/run-config"
mkdir -p "$RUNTIME_DIR"

# Emit authSettings.py using container env with safe defaults
cat > "$RUNTIME_DIR/authSettings.py" <<'PYCONF'
# -*- coding: utf-8 -*-
import os

# Neo4j Database Configuration
graphUser = os.getenv("GRAPH_USER", "neo4j")
graphPass = os.getenv("GRAPH_PASS", "neo4j")
graphURL  = os.getenv("GRAPH_URL", "bolt://polinom.uniri.hr:7687")

# Sketch Engine API Configuration
userName = os.getenv("SKETCH_USER", "your-sketch-engine-username")
apiKey   = os.getenv("SKETCH_API_KEY", "your-sketch-engine-api-key")
PYCONF

# Prepend runtime dir to PYTHONPATH so our generated module overrides the baked one
export PYTHONPATH="$RUNTIME_DIR:${PYTHONPATH:-}"

exec "$@"


