#!/bin/bash

if [ ! -d "${1}/KTH" ]
then
  mkdir -p ${1}/KTH/
fi

cd  ${1}/KTH/
wget http://www.nada.kth.se/cvap/actions/walking.zip
wget http://www.nada.kth.se/cvap/actions/jogging.zip
wget http://www.nada.kth.se/cvap/actions/running.zip
wget http://www.nada.kth.se/cvap/actions/boxing.zip
wget http://www.nada.kth.se/cvap/actions/handwaving.zip
wget http://www.nada.kth.se/cvap/actions/handclapping.zip

unzip walking.zip
unzip jogging.zip
unzip running.zip
unzip boxing.zip
unzip handwaving.zip
unzip handclapping.zip

rm walking.zip
rm jogging.zip
rm running.zip
rm boxing.zip
rm handwaving.zip
rm handclapping.zip
