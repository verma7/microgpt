#!/bin/bash
> scale2_results.txt
run() { name=$1; shift; for seed in 42 43; do r=$(./microgpt_v5 "$@" --seed $seed < names.txt | grep RESULT); echo "$name seed=$seed $r" >> scale2_results.txt; done }
FNG="--gains --finalnorm"
run e16_fng $FNG
run e32_fng     --embd 32 --hidden 128 --steps 30000 $FNG
run e32_base_lr5 --embd 32 --hidden 128 --steps 30000 --lr 0.005
run e32_fng_b8  --embd 32 --hidden 128 --steps 8000 --batch 8 $FNG
run e32_modg    --embd 32 --hidden 85 --steps 30000 --gains --finalnorm --tie --rope --mlp swiglu --residscale
run e64_fng     --embd 64 --heads 8 --hidden 256 --steps 30000 $FNG
run e64_fng_b8  --embd 64 --heads 8 --hidden 256 --steps 8000 --batch 8 $FNG
echo SCALE2_DONE
