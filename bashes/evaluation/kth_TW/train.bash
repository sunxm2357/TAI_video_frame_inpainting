#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=kth_TW_exp1 \
    --K=5 \
    --F=5 \
    --T=5 \
    --max_iter=100000 \
    --data=KTH --dataroot=${1} --textroot=videolist/KTH/ --c_dim=1 \
    --model=simplecomb --comb_loss=ToTarget --comb_type=w_avg \
    --image_size 128 \
    --final_sup_update=1 \
    --batch_size=8 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --Ip=3 \
    --sn \
    --enable_res \
    ${RESUME}
