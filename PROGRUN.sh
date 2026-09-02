#!/usr/bin/env bash

if [[ ! -f "./data/datarand.bin" || ! -f "./data/datacond.bin" ]]; then
    ./bin/datagen
fi

# ./bin/main
