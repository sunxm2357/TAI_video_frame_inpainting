#!/bin/bash

if [ "${2}" == "resume" ]
then
    RESUME='--continue_train'
else
    RESUME=' '
fi

python src/train.py  \
    --name=hmdb_TW_exp1 \
    --K=4 \
    --F=4 \
    --T=3 \
    --max_iter=200000 \
    --data=HMDB51 --dataroot=${1} --textroot=videolist/HMDB/ --c_dim=3  \
    --model=simplecomb --comb_loss=ToTarget --comb_type=w_avg \
    --image_size 160 208 \
    --gpu_ids 0 \
    --final_sup_update=1 \
    --batch_size=4 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --Ip=3 \
    --sn \
    --enable_res \
    ${RESUME}
