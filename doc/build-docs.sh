#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Skigen Contributors
#
# build-docs.sh — Generate API reference Markdown for Docusaurus.
# Usage: cd doc && bash build-docs.sh
#
# Requires:
#   - doxygen  (https://www.doxygen.nl/)
#   - python3

set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1/2] Running Doxygen (SkigenPlot) ==="
doxygen Doxyfile

echo "=== [2/2] Copying guide pages ==="
mkdir -p website/docs/guide
cp -r guide/*.mdx website/docs/guide/ 2>/dev/null || true

echo "=== Done ==="
echo "Run 'cd website && npm run build' to build the full site."
