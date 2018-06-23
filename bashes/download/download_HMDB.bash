#!/bin/bash

if [ ! -d "${1}/HMDB-51" ]
then
  mkdir -p ${1}/HMDB-51/
fi

cd  ${1}/HMDB-51/
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar
rm hmdb51_org.rar
for f in *.rar; do
	unrar x $f
	rm $f
done