#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=debug_kth_TAI_exp1 \
    --K=5 \
    --F=5 \
    --T=5 \
    --max_iter=200000 \
    --data=KTH --dataroot=${1} --textroot=videolist/KTH/ --c_dim=1 \
    --model=kernelcomb --comb_loss=ToTarget --comb_type=avg \
    --image_size 128 \
    --final_sup_update=1 \
    --batch_size=1 \
    --num_block=5 \
    --layers=3 \
    --kf_dim=32 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --rc_loc=4 \
    --Ip=3 \
    --display_freq=1 \
    --print_freq=1 \
    --save_latest_freq=1 \
    --validate_freq=1 \
    --sn \
    --enable_res \
    --shallow \
    ${RESUME}

