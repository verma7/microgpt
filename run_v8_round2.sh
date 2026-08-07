#!/bin/bash
> v8_r2_results.txt
BASE="--threads 4 --steps 2000 --eval-every 1000 --nofinal --rope --tie --finalnorm --gains --residscale --dwconv --softmax1"
run() { name=$1; shift; ./microgpt_v8 $BASE "$@" > v8_$name.log 2>&1; echo "$name | $(grep 'tok/s' v8_$name.log | tail -1)" >> v8_r2_results.txt; }
run muon_lr01     --mlp swiglu --hidden 344 --muon --muonlr 0.01
run muon_lr04     --mlp swiglu --hidden 344 --muon --muonlr 0.04
run muon_relu2    --mlp relu2 --hidden 512 --muon --muonlr 0.02
run muon_relu2_ve --mlp relu2 --hidden 512 --muon --muonlr 0.02 --valemb
echo V8_ROUND2_DONE
