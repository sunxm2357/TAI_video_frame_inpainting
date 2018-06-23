#!/bin/bash

python src/test.py \
    --name=ucf_mcnet_exp1 \
    --K=${1} \
    --T=${2} \
    --data=HMDB51 --dataroot=${3} --textroot=videolist/HMDB/ --c_dim=3 --pick_mode First \
    --model=mcnet \
    --image_size 160 208 \
    --batch_size=2 \
    --pick_mode=First \
    --enable_res \
