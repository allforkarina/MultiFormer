#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

for env in E01 E02 E03 E04; do
  echo "=== Training $env ==="
  python train.py --config "configs/${env}.yaml"
done
