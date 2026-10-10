#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

N_SAMPLES=1000000

# Checks if all files exist
data_ok() {
    local kind f path expected
    for kind in well ill; do
        for f in $(seq 11 10 101); do
            path="data/${kind}cond/data${kind}${f}.bin"
            expected=$(( 16 + 8 * (f + N_SAMPLES + N_SAMPLES * f) ))

            [[ -f "$path" && "$(stat -c %s "$path")" -eq "$expected" ]] \
                || return 1
        done
    done
}

make

if ! data_ok; then
    make datagen
    data_ok || { echo "Invalid data after datagen" >&2; exit 1; }
fi

mkdir -p data/results

export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=4294967296
/usr/bin/time -v -o data/results/$(date +"%Y%m%d-%H%M%S")-time.log ./bin/main

