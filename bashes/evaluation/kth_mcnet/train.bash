#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=kth_mcnet_exp1 \
    --K=5 \
    --T=5 \
    --max_iter=200000 \
    --data=KTH --dataroot=${1} --textroot=videolist/KTH/ --c_dim=1 \
    --model=mcnet \
    --image_size 128 \
    --batch_size=8 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --Ip=3 \
    --sn \
    --enable_res \
    ${RESUME}
