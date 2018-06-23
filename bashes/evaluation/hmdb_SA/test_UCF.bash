#!/bin/bash

python src/test.py \
    --name=hmdb_SA_exp1 \
    --K=${1} \
    --F=${1}  \
    --T=${2} \
    --data=UCF --dataroot=${3} --textroot=videolist/UCF/ --c_dim=3 --pick_mode=First \
    --model=simplecomb --comb_type=avg \
    --image_size 160 208 \
    --batch_size=2 \
    --pick_mode=First \
    --enable_res \
    --output_both_directions \
