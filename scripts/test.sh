#!/usr/bin/env sh
# The same checks CI runs: unit tests (with the backbone parity test), the Embedded Swift WebAssembly build,
# and the WebAssembly/native parity check.
set -e
cd "$(dirname "$0")/.."
./scripts/get-ort
./scripts/swift test
./scripts/build-wasm
node scripts/check-wasm.mjs
