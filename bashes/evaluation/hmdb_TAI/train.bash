#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=hmdb_TAI_exp1 \
    --K=4 \
    --F=4 \
    --T=3 \
    --max_iter=200000 \
    --data=HMDB51 --dataroot=${1} --textroot=videolist/HMDB/ --c_dim=3  \
    --model=kernelcomb --comb_loss=ToTarget --comb_type=avg \
    --image_size 160 208 \
    --gpu_ids 0 \
    --final_sup_update=1 \
    --batch_size=4 \
    --num_block=4 \
    --layers=3 \
    --kf_dim=32 \
    --rc_loc=4 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --Ip=3 \
    --sn \
    --enable_res \
    --shallow \
    ${RESUME}
