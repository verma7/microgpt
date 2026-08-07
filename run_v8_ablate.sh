#!/bin/bash
> v8_results.txt
BASE="--threads 4 --steps 2000 --eval-every 1000 --nofinal --rope --tie --finalnorm --gains --residscale --dwconv --softmax1"
run() { name=$1; shift; ./microgpt_v8 $BASE "$@" > v8_$name.log 2>&1; echo "$name | $(grep 'tok/s' v8_$name.log | tail -1)" >> v8_results.txt; }
run base     --mlp swiglu --hidden 344
run muon     --mlp swiglu --hidden 344 --muon --muonlr 0.02
run relu2    --mlp relu2 --hidden 512
run valemb   --mlp swiglu --hidden 344 --valemb
run softcap  --mlp swiglu --hidden 344 --softcap 15
echo V8_ABLATE_DONE
