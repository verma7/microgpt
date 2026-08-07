#!/bin/bash
declare -a NAMES=(base2k base fn fn_tie fn_rope fn_tie_rope gelu swiglu swiglu_rs)
declare -a FLAGS=(
  "--steps 2000"
  ""
  "--finalnorm"
  "--finalnorm --tie"
  "--finalnorm --rope"
  "--finalnorm --tie --rope"
  "--finalnorm --tie --rope --mlp gelu"
  "--finalnorm --tie --rope --mlp swiglu --hidden 43"
  "--finalnorm --tie --rope --mlp swiglu --hidden 43 --residscale"
)
> ablation_results.txt
for i in "${!NAMES[@]}"; do
  for seed in 42 43 44; do
    r=$(./microgpt_v5 ${FLAGS[$i]} --seed $seed < names.txt | grep RESULT)
    echo "${NAMES[$i]} seed=$seed $r" >> ablation_results.txt
  done
done
echo ABLATION_DONE
