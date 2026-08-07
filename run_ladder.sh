#!/bin/bash
COMMON="--threads 8 --eval-every 2000 --rope --tie --finalnorm --gains --residscale --mlp swiglu --dwconv --softmax1 --muon --muonlr 0.02 --wd 0.05 --clip 1.0 --evalstride 128"
./microgpt_v8 $COMMON --embd 256 --layers 8 --heads 8 --block 512 --hidden 688 --batch 16 --steps 40000 --lr 7e-4 --warmup 1000 --save enwik8_r2b.bin > enwik8_r2b.log 2>&1
echo RUNG2B_DONE
./microgpt_v8 $COMMON --embd 384 --layers 10 --heads 12 --block 512 --hidden 1024 --batch 16 --steps 60000 --lr 5e-4 --warmup 1500 --save enwik8_r3.bin > enwik8_r3.log 2>&1
echo RUNG3_DONE
