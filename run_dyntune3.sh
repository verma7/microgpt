#!/bin/bash
> dyntune3_results.txt
./microgpt_v8d --threads 4 --steps 0 --load enwik8_r4.bin --evalstride 512 --dyneval 1e-5 --dynbatch 4 --dynvalonly --rope --tie --finalnorm --gains --residscale --mlp swiglu --dwconv --softmax1 --embd 384 --layers 10 --heads 12 --block 1024 --hidden 1024 2>/dev/null | grep DYNRESULT >> dyntune3_results.txt
./microgpt_v8d --threads 4 --steps 0 --load enwik8_r4.bin --evalstride 512 --dyneval 2e-5 --dynbatch 2 --dynvalonly --rope --tie --finalnorm --gains --residscale --mlp swiglu --dwconv --softmax1 --embd 384 --layers 10 --heads 12 --block 1024 --hidden 1024 2>/dev/null | grep DYNRESULT >> dyntune3_results.txt
echo DYNTUNE3_DONE
