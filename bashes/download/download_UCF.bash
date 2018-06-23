#!/bin/bash

if [ ! -d "${1}/UCF-101" ]
then
  mkdir -p ${1}/UCF-101/
fi

cd ${1}/UCF-101/

wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar
rm UCF101.rar

mv UCF-101/* .
rm -r UCF-101