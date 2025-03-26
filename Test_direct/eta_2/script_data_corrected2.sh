#!/bin/bash


##
## ADD AFTER AN ADDITIONAL SCRIPT TO FIND THE COLUMNS CORRESPONDING TO THE LOWST
## IMAGINARY ENERGY, INSTEAD OF SELECTING THEM IN THE CODE
##

#awk {'print "%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", $1, $2, $3, $4, $5, -$4/1, -$5/1, $1*sqrt(($4/$1)^2+($5/$1)^2)'} datafile.dat
awk '{printf "%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", $1, $2, $3, $4, $5, -$4/$1, -$5/$1, $1*sqrt(($4/$1)^2+($5/$1)^2)}' $1
