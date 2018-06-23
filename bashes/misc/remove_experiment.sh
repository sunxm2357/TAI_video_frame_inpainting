#!/bin/bash

# 1 is experiment name
# 2 is dataset name

CHECKPOINT_DIR=checkpoints/${1}
TB_DIR=tb/${1}
RESULT_IMAGE_DIR=results/images/${2}/${1}_*
RESULT_QUANT_DIR=results/quantitative/${2}/${1}_*

if [ -d "$CHECKPOINT_DIR" ];
then
    echo "deleting checkpoints"
    rm -r $CHECKPOINT_DIR
fi

if [ -d "$TB_DIR" ];
then
    echo "deleting tensorboard file"
    rm -r $TB_DIR
fi

if [ -d "$RESULT_IMAGE_DIR" ];
then
    echo "deleting qualitative result"
    rm -r $RESULT_IMAGE_DIR
fi

if [ -d "$RESULT_QUANT_DIR" ];
then
    echo "deleting quantitative result"
    rm -r $RESULT_QUANT_DIR
fi
