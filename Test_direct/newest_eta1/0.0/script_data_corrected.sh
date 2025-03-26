#!/bin/bash


##
## ADD AFTER AN ADDITIONAL SCRIPT TO FIND THE COLUMNS CORRESPONDING TO THE LOWST
## IMAGINARY ENERGY, INSTEAD OF SELECTING THEM IN THE CODE
##
# Extract 'CAP energies' data and format it with 10 decimal places

grep -A$(( $2+10 )) 'CAP energies' "$1" | tail -"$2" | awk '{
    if ($1 != 0)
        printf "%.10f %.10f %.10f %.10f %.10f\n", $1, $2, $3, $4, $5;
    else
        printf "%.10f %.10f %.10f %.10f %.10f\n",0.0000000000, $2, 0.0000000000, $4, 0.0000000000
}' > "$1.data"
#awk '{printf "%.10f %.10f %.10f\n", $1, $4, $5}' > "$1.data"

# Extract 'Energy corrections' data and compute additional columns
grep -A$(( $2+7 )) 'Energy corrections' "$1" | tail -"$2" | awk '{
    if ($1 != 0)
        printf "%.10f %.10f %.10f %.10f\n", $2, $3, $4, $5;
    else
        printf "%.10f %.10f %.10f %.10f\n", $2, 0.0000000000, $4, 0.0000000000
}' > "$1.corrections.data"
#awk '{printf "%.10f %.10f %.10f %.10f %.10f\n", $4, $5, -$4/$1, -$5/$1, $1*sqrt(($4/$1)^2+($5/$1)^2)}' > "$1.corrections.data"

# Merge the two output files
paste "$1.data" "$1.corrections.data" > "$1.data$3.good"

# Clean up temporary files
rm -f "$1.data" "$1.corrections.data"

python3.11 Good_columns.py $1.data$3.good temp1

awk '{
    if ($1 != 0)
        printf "%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", $1, $2, $3, $6, $7, -$6/$1, -$7/$1, $1*sqrt(($6/$1)^2+($7/$1)^2);
    else
        printf "%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", $1, $2, 0.00000, 0.00000, 0.0000000, 0.0000000, 0.00000000, 0.0000000000
}' temp1 > "$1.data$3.good"

#rm -f temp1
#grep -A$(( $2+10 )) 'CAP energies' "$1" | tail -"$2" | awk '{
#    if ($1 != 0) 
#        printf "%.10f %.10f %.10f\n", $1, $4, -$5;
#    else 
#        printf "0.0000000000 0.0000000000 0.0000000000\n"
#}' > "$1.data"
#	#awk '{printf "%.10f %.10f %.10f\n", $1, $4, $5}' > "$1.data"
#
## Extract 'Energy corrections' data and compute additional columns
#grep -A$(( $2+7 )) 'Energy corrections' "$1" | tail -"$2" | awk '{
#    if ($1 != 0) 
#        printf "%.10f %.10f %.10f %.10f %.10f\n", $4, $5, -$4/$1, -$5/$1, $1*sqrt(($4/$1)^2+($5/$1)^2);
#    else 
#        printf "0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000\n"
#}' > "$1.corrections.data"
#	#awk '{printf "%.10f %.10f %.10f %.10f %.10f\n", $4, $5, -$4/$1, -$5/$1, $1*sqrt(($4/$1)^2+($5/$1)^2)}' > "$1.corrections.data"
#
## Merge the two output files
#paste "$1.data" "$1.corrections.data" > "$1.data$3.good"
#
## Clean up temporary files
#rm -f "$1.data" "$1.corrections.data"
