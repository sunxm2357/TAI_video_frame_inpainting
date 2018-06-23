#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=kth_TWI_exp1 \
    --K=5 \
    --F=5 \
    --T=5 \
    --max_iter=200000 \
    --data=KTH --dataroot=${1} --textroot=videolist/KTH/ --c_dim=1 \
    --model=kernelcomb --comb_loss=ToTarget --comb_type=w_avg \
    --image_size 128 \
    --final_sup_update=1 \
    --batch_size=4 \
    --num_block=5 \
    --layers=3 \
    --kf_dim=32 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --Ip=3 \
    --sn \
    --enable_res \
    --shallow \
    ${RESUME}
