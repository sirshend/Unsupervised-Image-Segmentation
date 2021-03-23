#!/bin/bash

mkdir output
mkdir output2

### Runs the testbg.py script for every camera directory. 
### This gives the number of objects/clusters in the image 
for i in {1..5}
do
	echo $i
	python3 testbg.py $i >> cam.txt
done


### Gets the number of clusters outputted by testbg.py script and then runs autoseg.py for every image
### This autoseg.py script does the actual segmentation 
./code.py cam.txt | xargs -n3 bash -c './autoseg.py --camera=$0 --input=$1 --minLabels=$2 >> result.txt'
