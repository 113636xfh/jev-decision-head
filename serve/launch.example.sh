#!/bin/sh
# Example only: how the patched server is started, with placeholders.
# Our exact launch configuration is pinned in REPRODUCE.md §0a. Apply patches/ first (see patches/README.md
# and patches/manifest.json for the pinned upstream commit), then record an identity:
#   python -m jev.identity --out identity/runtime-identity.json
set -eu
STRATA_RUNTIME=${STRATA_RUNTIME:-/opt/strata/v0.1.44.1}
CONFIG=${JEV_SERVER_CONFIG:-$STRATA_RUNTIME/strata-iq3_s.json}
# Required for the readout path: the Jev readout capability must be advertised.
exec "$STRATA_RUNTIME/engine-cuda12/strata" --config "$CONFIG" \
  --batch-slots 4 --batch-groups 2 --max-context 262144
