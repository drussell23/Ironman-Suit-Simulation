#!/usr/bin/env bash
set -euo pipefail

# List of files to ensure have a trailing newline
FILES="
  include/aerodynamics/mesh.h
  include/aerodynamics/turbulence_model.h
  include/aerodynamics/actuator.h
  include/aerodynamics/flow_state.h
  src/aerodynamics/actuator.c
  src/aerodynamics/flow_state.c
  src/aerodynamics/mesh.c
"

echo "⟳ Running EOF‐newline fixup..."

for f in $FILES; do
  if [ ! -f "$f" ]; then
    echo "  ⚠️  Skipping missing file: $f"
    continue
  fi

  # Grab the last byte; if it's neither empty nor a newline, append one
  last=$(tail -c 1 "$f" || true)
  if [ -n "$last" ] && [ "$last" != $'\n' ]; then
    printf '\n' >> "$f"
    echo "  ✏️  Appended newline to: $f"
  else
    echo "  ✔️  Already OK:      $f"
  fi
done

echo "✅ EOF‐newline fixup complete."
