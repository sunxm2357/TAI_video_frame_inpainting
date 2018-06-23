#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=ucf_mcnet_exp1 \
    --K=4 \
    --T=3 \
    --max_iter=200000 \
    --data=UCF --dataroot=${1} --textroot=videolist/UCF/ --c_dim=3 \
    --model=mcnet \
    --image_size 160 208 \
    --gpu_ids 0 \
    --final_sup_update=1 \
    --batch_size=8 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --Ip=3 \
    --sn \
    --enable_res \
    ${RESUME}
