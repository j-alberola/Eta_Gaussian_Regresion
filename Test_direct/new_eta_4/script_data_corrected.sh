#!/bin/bash


##
## ADD AFTER AN ADDITIONAL SCRIPT TO FIND THE COLUMNS CORRESPONDING TO THE LOWST
## IMAGINARY ENERGY, INSTEAD OF SELECTING THEM IN THE CODE
##
# Extract 'CAP energies' data and format it with 10 decimal places
grep -A$(( $2+10 )) 'CAP energies' "$1" | tail -"$2" | awk '{printf "%.10f %.10f %.10f\n", $1, $4, $5}' > "$1.data"

# Extract 'Energy corrections' data and compute additional columns
grep -A$(( $2+7 )) 'Energy corrections' "$1" | tail -"$2" | awk '{printf "%.10f %.10f %.10f %.10f %.10f\n", $4, $5, -$4/$1, -$5/$1, $1*sqrt(($4/$1)^2+($5/$1)^2)}' > "$1.corrections.data"

# Merge the two output files
paste "$1.data" "$1.corrections.data" > "$1.data$3.good"

# Clean up temporary files
rm -f "$1.data" "$1.corrections.data"
