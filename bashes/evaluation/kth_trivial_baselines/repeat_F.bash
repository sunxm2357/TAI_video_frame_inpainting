#!/bin/bash

python src/trivial_test.py  \
    --name=kth_repeat_F \
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --c_dim=1 --pick_mode=Slide \
    --comb_type=repeat_F \
    --image_size 128 \
