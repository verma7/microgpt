#!/bin/bash
> scale_results.txt
run() { name=$1; shift; for seed in 42 43; do r=$(./microgpt_v5 "$@" --seed $seed < names.txt | grep RESULT); echo "$name seed=$seed $r" >> scale_results.txt; done }
MOD="--finalnorm --tie --rope --mlp swiglu --residscale"
run e32_base --embd 32 --hidden 128 --steps 30000
run e32_mod  --embd 32 --hidden 85  --steps 30000 $MOD
run e64_base --embd 64 --heads 8 --hidden 256 --steps 30000
run e64_mod  --embd 64 --heads 8 --hidden 171 --steps 30000 $MOD
echo SCALE_DONE
