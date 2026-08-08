#!/bin/bash
> dyntune2_results.txt
for lr in 2e-5 5e-5; do
  ./microgpt_v8d --threads 4 --steps 0 --load enwik8_r4.bin --evalstride 512 --dyneval $lr --dynbatch 4 --dynvalonly --rope --tie --finalnorm --gains --residscale --mlp swiglu --dwconv --softmax1 --embd 384 --layers 10 --heads 12 --block 1024 --hidden 1024 2>/dev/null | grep DYNRESULT >> dyntune2_results.txt
done
echo DYNTUNE2_DONE
