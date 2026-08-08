#!/bin/bash
> dyntune_results.txt
for lr in 5e-5 1.5e-4 4e-4; do
  ./microgpt_v8d --threads 4 --steps 0 --load enwik8_r4.bin --evalstride 0 --dyneval $lr --rope --tie --finalnorm --gains --residscale --mlp swiglu --dwconv --softmax1 --embd 384 --layers 10 --heads 12 --block 1024 --hidden 1024 --batch 16 2>/dev/null | grep -E "^(RESULT|DYNRESULT)" >> dyntune_results.txt
done
echo DYNTUNE_DONE
