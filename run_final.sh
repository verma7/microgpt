#!/bin/bash
> final_results.txt
run() { name=$1; shift; for seed in "${SEEDS[@]}"; do r=$(./microgpt_v5 "$@" --seed $seed < names.txt | grep RESULT); echo "$name seed=$seed $r" >> final_results.txt; done }
SEEDS=(42 43)
run e32_base_b8_16k --embd 32 --hidden 128 --steps 16000 --batch 8
SEEDS=(42)
run e32_base_b8_32k --embd 32 --hidden 128 --steps 32000 --batch 8
run e32_fng_b8_32k  --embd 32 --hidden 128 --steps 32000 --batch 8 --gains --finalnorm
echo FINAL_DONE
