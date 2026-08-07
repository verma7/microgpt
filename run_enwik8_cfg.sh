#!/bin/bash
> enwik8_cfg_results.txt
run() { name=$1; shift; ./microgpt_v6 --threads 8 --steps 2000 --eval-every 500 --nofinal "$@" > cfg_$name.log 2>&1; echo "$name $(grep 'tok/s' cfg_$name.log | tail -1)" >> enwik8_cfg_results.txt; }
run plain
run mid    --finalnorm --gains --residscale
run modern --rope --tie --finalnorm --gains --residscale --mlp swiglu --hidden 344
echo CFG_DONE
