#!/bin/bash

python src/test.py \
    --name=kth_TAI_exp1 \
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --c_dim=1 --pick_mode=Slide \
    --model=kernelcomb --comb_type=avg \
    --image_size 128 \
    --batch_size=1 \
    --num_block=5 \
    --layers=3 \
    --kf_dim=32 \
    --rc_loc=4 \
    --enable_res \
    --output_both_directions \
    --shallow \
