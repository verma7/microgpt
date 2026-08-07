#!/bin/bash
> scale3_results.txt
run() { name=$1; shift; for seed in "${SEEDS[@]}"; do r=$(./microgpt_v5 "$@" --seed $seed < names.txt | grep RESULT); echo "$name seed=$seed $r" >> scale3_results.txt; done }
FNG="--gains --finalnorm"
SEEDS=(42 43)
run e32_base_b8    --embd 32 --hidden 128 --steps 8000 --batch 8
run e32_fng_b8_16k --embd 32 --hidden 128 --steps 16000 --batch 8 $FNG
run e32_fng_b8_lr2 --embd 32 --hidden 128 --steps 8000 --batch 8 --lr 0.02 $FNG
run e48_fng_b8     --embd 48 --heads 6 --hidden 192 --steps 8000 --batch 8 $FNG
SEEDS=(44)
run e32_fng_b8     --embd 32 --hidden 128 --steps 8000 --batch 8 $FNG
echo SCALE3_DONE
