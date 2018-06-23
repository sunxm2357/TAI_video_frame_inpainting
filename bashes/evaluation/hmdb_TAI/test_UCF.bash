#!/bin/bash

python src/test.py \
    --name=hmdb_TAI_exp1 \
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=UCF --dataroot=${3} --textroot=videolist/UCF/ --c_dim=3 --pick_mode=First \
    --model=kernelcomb  --comb_type=avg \
    --image_size 160 208 \
    --batch_size=2 \
    --num_block=4 \
    --layers=3 \
    --kf_dim=32 \
    --pick_mode=First \
    --rc_loc=4 \
    --enable_res \
    --output_both_directions \
    --shallow \
