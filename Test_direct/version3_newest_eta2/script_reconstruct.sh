#!/bin/bash

for i in {2..14}; do

	mkdir $i
	cp N2_minus.hf.4states.no.cipsi.17200.data1.good $i
	cp PROGRAM.py $i
	cp PROGRAM_test.py $i
	cd $i 
        head -n $i N2_minus.hf.4states.no.cipsi.17200.data1.good > temp_file
	mv temp_file N2_minus.hf.4states.no.cipsi.17200.data1.good
	sort -g -k1,1 N2_minus.hf.4states.no.cipsi.17200.data1.good > datafile.dat
        cd ..
done
