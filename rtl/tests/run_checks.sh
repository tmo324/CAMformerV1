#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

for tool in iverilog vvp verilator; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing required RTL tool: $tool" >&2
        exit 2
    fi
done

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/camformer-rtl.XXXXXX")"
cleanup() {
    rm -rf "$build_dir"
}
trap cleanup EXIT

iverilog -g2012 -tnull -f rtl/filelists/iverilog.f

iverilog -g2012 \
    -s tb_q_buffer \
    -o "$build_dir/tb_q_buffer.vvp" \
    rtl/a1_q_buffer.sv \
    rtl/tests/tb_q_buffer.sv
vvp "$build_dir/tb_q_buffer.vvp"

iverilog -g2012 \
    -s tb_adc \
    -o "$build_dir/tb_adc.vvp" \
    rtl/a4_9bit_adc.sv \
    rtl/tests/tb_adc.sv
vvp "$build_dir/tb_adc.vvp"

verilator --lint-only --timing -Wall -Wno-fatal -f rtl/filelists/lint.f

echo "PASS RTL syntax, smoke tests, and lint"
