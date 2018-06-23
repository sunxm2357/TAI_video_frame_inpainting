#!/bin/bash

python src/test.py \
    --name=kth_mcnet_exp1 \
    --K=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --c_dim=1 --pick_mode=Slide \
    --model=mcnet \
    --image_size 128 \
    --batch_size=1 \
    --enable_res \
