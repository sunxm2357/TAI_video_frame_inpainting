#!/bin/bash

python src/trivial_test.py  \
    --name=kth_TW_P_F\
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --c_dim=1 --pick_mode=Slide \
    --comb_type=w_avg \
    --image_size 128 \
