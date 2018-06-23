#!/bin/bash

python src/test.py \
    --name=ucf_TW_exp1 \
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=HMDB51 --dataroot=${3} --textroot=videolist/HMDB/ --c_dim=3 --pick_mode=First \
    --model=simplecomb --comb_type=w_avg \
    --image_size 160 208 \
    --batch_size=2 \
    --pick_mode=First \
    --enable_res \
    --output_both_directions \
