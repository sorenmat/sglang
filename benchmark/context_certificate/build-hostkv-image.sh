#!/bin/bash
# Build hostkv:4ccff141-1 from the host-kv/main checkout (HKV-21).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sudo -n docker build -f docker/Dockerfile.hostkv -t hostkv:4ccff141-1 .
sudo -n docker image inspect hostkv:4ccff141-1 --format "built {{.Id}}"
