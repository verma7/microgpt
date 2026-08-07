#!/bin/bash
> novel_results.txt
BASE="--threads 8 --steps 2000 --eval-every 1000 --nofinal --rope --tie --finalnorm --gains --residscale --mlp swiglu --hidden 344"
run() { name=$1; shift; ./microgpt_v7 $BASE "$@" > novel_$name.log 2>&1; echo "$name | $(grep 'tok/s' novel_$name.log | tail -1) | $(grep 'params=' novel_$name.log | head -1 | grep -o 'params=[0-9]*')" >> novel_results.txt; }
run modernbase
run dwconv   --dwconv
run valres   --valres
run attngate --attngate
run qknorm   --qknorm
run softmax1 --softmax1
echo NOVEL_DONE
