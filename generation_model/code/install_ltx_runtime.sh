#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNTIME_ROOT="${SCRIPT_DIR}/ltx_v2"
RUNTIME_ROOT="${LTX_V2_ROOT:-${DEFAULT_RUNTIME_ROOT}}"
UPSTREAM="https://github.com/Lightricks/LTX-2.git"
REVISION="d887bbd1e045b316d0a855f8d421209ecfc97ed1"

command -v git >/dev/null 2>&1 || { echo "git is required to fetch LTX-2." >&2; exit 2; }

if [[ -e "${RUNTIME_ROOT}" ]]; then
  echo "Runtime destination already exists: ${RUNTIME_ROOT}" >&2
  echo "Choose another empty LTX_V2_ROOT or move the existing directory yourself." >&2
  exit 2
fi

mkdir -p "$(dirname -- "${RUNTIME_ROOT}")"
git init --quiet "${RUNTIME_ROOT}"
git -C "${RUNTIME_ROOT}" remote add origin "${UPSTREAM}"
git -C "${RUNTIME_ROOT}" fetch --quiet --depth 1 origin "${REVISION}"
git -C "${RUNTIME_ROOT}" checkout --quiet --detach FETCH_HEAD

actual_revision="$(git -C "${RUNTIME_ROOT}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${REVISION}" ]]; then
  echo "Unexpected LTX-2 revision: ${actual_revision}" >&2
  exit 2
fi

echo "Installed LTX-2 ${REVISION} at ${RUNTIME_ROOT}"
echo "The inference entry point uses the upstream TI2VidOneStagePipeline."
echo "Review the upstream license at ${RUNTIME_ROOT}/LICENSE before using the model or adapter."
