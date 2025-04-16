#!/usr/bin/env qpsh

#SBATCH -J bu.p
#SBATCH -p multi -c 30  -n 1 -N 1
#SBATCH --mem=50000

##MODIFIY FORTRAN SCRIPT TO WRITE OUTPUT IN A GIVEN FILE



module load intel
module load lcpq/gnu/gcc-7.5.0
source ~/PHD/qp2/quantum_package.rc


qp set_file $1
#qp edit -s [1,2]
qp set cap do_cap true
qp set cap eta_cap 0.000
qp set cap eta_step_size 0.0075
qp set cap n_steps_cap  2
qp set davidson_keywords n_det_max_full 1000
qp set davidson_keywords state_following false
qp set determinants read_wf True
#qp set determinants n_states 3

qp run diagonalize_h_cap > $1.scan.out
cp $1.scan.out $1.scan.script.1st.out
./script_data_corrected.sh $1.scan.out 2 1
\cp -f $1.scan.out.data1.good datafile.dat
python3.11 PROGRAM_bayesian_optimization_direct_CONSTRAINED_test_normalized.py

threshold_eta=0.000001
threshold_velocity=0.0000005
qp set cap eta_step_size 0.
qp set cap n_steps_cap  1

cp Optimal_LB Optimal_LB_original
cp Optimal_EI Optimal_EI_original
cp datafile.dat datafile_original.dat
#  ADD A LOOP HERE
#  MAYBE REMOVE THE DIFFERENT 2 CASES, JUST 1 IS ENOUGH

iter=1
while [ $iter -lt 30 ] ; do

    new_min=$(awk 'NR == 1 {print $1}' Optimal_LB)
    fit_velocity=$(awk 'NR == 1 {print $2}' Optimal_LB)
    PDF_maximum=$(awk 'NR == 1 {print $3}' Optimal_LB)
    if awk "BEGIN {exit !($new_min == 0)}"; then
    #if [ "$new_min" -eq 0 ]; then
        # Get the lowest point after 0 in datafile.dat
        new_min=$(awk '$1 > 0 {print $1}' datafile.dat | sort -g | head -n 1)
        new_min=$(awk "BEGIN {print $new_min / 2}")
    fi

    qp set cap eta_cap $new_min
    qp run diagonalize_h_cap > $1.scan.LB.out
    cp $1.scan.LB.out $1.scan.script.$new_min.LB.out
    ./script_data_corrected.sh $1.scan.LB.out 1 2
    echo $new_min > min_file
    awk {'print $2, $3, $4, $5, $6, $7, $8'} $1.scan.LB.out.data2.good > energy_file
    paste min_file energy_file >> $1.scan.out.data1.good
    sort -g -k1,1 $1.scan.out.data1.good > $1.data1.LB.good.sorted
    \cp -f $1.data1.LB.good.sorted datafile.dat
    qp_velocity=$(awk 'NR == 1 {print $8}' $1.scan.LB.out.data2.good)
    velocity_diff=$(awk "BEGIN {print $qp_velocity - $fit_velocity}")
    if [ $(awk "BEGIN {if ($velocity_diff < 0) print 1; else print 0}") -eq 1 ]; then
    	velocity_diff=$(awk "BEGIN {print -1*$velocity_diff}")
    fi
    python3.11 PROGRAM_bayesian_optimization_direct_CONSTRAINED_test_normalized.py
    new_min2=$(awk 'NR == 1 {print $1}' Optimal_LB)
    eta_diff=$(awk "BEGIN {print $new_min - $new_min2}")
    if [ $(awk "BEGIN {if ($eta_diff < 0) print 1; else print 0}") -eq 1 ]; then
    	eta_diff=$(awk "BEGIN {print -1*$eta_diff}")
    fi
    echo 'Difference in velocity:'$velocity_diff
    echo 'Difference in eta:'$eta_diff
    echo 'Max PDF'$PDF_maximum

#
#   SECTION TO PRINT IF THE CRITERIA FOR ALL THE CONVERGENCE CRITERIA HAS BEEN FULFILLED
#

    v_flag=$(awk "BEGIN {print ($velocity_diff < $threshold_velocity) ? \"yes\" : \"no\"}")
    eta_flag=$(awk "BEGIN {print ($eta_diff < $threshold_eta) ? \"yes\" : \"no\"}")
    pdf_flag=$(awk "BEGIN {print ($PDF_maximum > 70) ? \"yes\" : \"no\"}")
    
    # Append flags to the last line of the file
#    total_lines=$(wc -l < "$1.scan.out.data1.good")
#    head -n $((total_lines - 1)) "$1.scan.out.data1.good" > tmp_file
    tail -n 1 "$1.scan.out.data1.good" | awk -v vf="$v_flag" -v ef="$eta_flag" -v pf="$pdf_flag" '{print $0, vf, ef, pf}' >> tmp_file
    
    # Overwrite original
    \cp tmp_file "$1.scan.out.data1.good.YES"
    
    # Sort final output
    sort -g -k1,1 "$1.scan.out.data1.good.YES" > "$1.data1.LB.good.sorted.YES"
    
    if [ $(awk "BEGIN {if (($velocity_diff < $threshold_velocity) && ($eta_diff < $threshold_eta) && ($PDF_maximum > 70)) print 1; else print 0}") -eq 1 ]; then
	qp set cap eta_cap $new_min2
    	qp run diagonalize_h_cap > $1.scan.LB.out
   	cp $1.scan.LB.out $1.scan.script.$new_min2.LB.out
   	./script_data_corrected.sh $1.scan.LB.out 1 2
   	echo $new_min2 > min_file
   	awk {'print $2, $3, $4, $5, $6, $7, $8'} $1.scan.LB.out.data2.good > energy_file
   	paste min_file energy_file >> $1.scan.out.data1.good
   	sort -g -k1,1 $1.scan.out.data1.good > $1.data1.LB.good.sorted
   	\cp -f $1.data1.LB.good.sorted datafile.dat
        echo $iter	
	cp datafile.dat datafile_LB.dat
        break
    fi
  
iter=$(($iter + 1))
done
cp $1.scan.out.data1.good $1.scan.out.data1.LB.good
rm -f datafile.dat
rm -f $1.scan.out.data1.good
awk -v iter="$iter" 'END {print $0, iter}' $1.scan.out.data1.LB.good > Last_point_LB
#awk 'END {print}, $iter' $1.scan.out.data1.LB.good > Last_point_LB



cp datafile_original.dat datafile.dat
cp datafile.dat $1.scan.out.data1.good
#iter=1
#while [ $iter -lt 30 ] ; do
#
#
#    if [ $iter -eq 1 ] ; then	
#    	new_min=$(awk 'NR == 1 {print $1}' Optimal_EI_original)
#    	fit_velocity=$(awk 'NR == 1 {print $2}' Optimal_EI_original)
#    else
#	new_min=$(awk 'NR == 1 {print $1}' Optimal_EI)
#        fit_velocity=$(awk 'NR == 1 {print $2}' Optimal_EI)
#    fi
#
#    qp set cap eta_cap $new_min
#    qp run diagonalize_h_cap > $1.scan.EI.out
#    cp $1.scan.EI.out $1.scan.script.$new_min.EI.out
#    ./script_data_corrected.sh $1.scan.EI.out 1 2
#    echo $new_min > min_file
#    awk {'print $2, $3, $4, $5, $6, $7, $8'} $1.scan.EI.out.data2.good > energy_file
#    paste min_file energy_file >> $1.scan.out.data1.good
#    sort -g -k1,1 $1.scan.out.data1.good > $1.data1.EI.good.sorted
#    \cp -f $1.data1.EI.good.sorted datafile.dat
#    qp_velocity=$(awk 'NR == 1 {print $8}' $1.scan.EI.out.data2.good)
#    velocity_diff=$(awk "BEGIN {print $qp_velocity - $fit_velocity}")
#    if [ $(awk "BEGIN {if ($velocity_diff < 0) print 1; else print 0}") -eq 1 ]; then
#        velocity_diff=$(awk "BEGIN {print -1*$velocity_diff}")
#    fi
#    python3.11 PROGRAM_bayesian_optimization_direct_test.py
#    new_min2=$(awk 'NR == 1 {print $1}' Optimal_EI)
#    eta_diff=$(awk "BEGIN {print $new_min - $new_min2}")
#    if [ $(awk "BEGIN {if ($eta_diff < 0) print 1; else print 0}") -eq 1 ]; then
#        eta_diff=$(awk "BEGIN {print -1*$eta_diff}")
#    fi
#    echo 'Difference in energy:'$velocity_diff
#    echo 'Difference in eta:'$eta_diff
#    if [ $(awk "BEGIN {if (($velocity_diff < $threshold_velocity) && ($eta_diff < $threshold_eta)) print 1; else print 0}") -eq 1 ]; then
#	qp set cap eta_cap $new_min2
#    	qp run diagonalize_h_cap > $1.scan.EI.out
#   	cp $1.scan.EI.out $1.scan.script.$new_min.EI.out
#   	./script_data_corrected.sh $1.scan.EI.out 1 2
#   	echo $new_min > min_file
#   	awk {'print $2, $3, $4, $5, $6, $7, $8'} $1.scan.EI.out.data2.good > energy_file
#   	paste min_file energy_file >> $1.scan.out.data1.good
#   	sort -g -k1,1 $1.scan.out.data1.good > $1.data1.EI.good.sorted
#   	\cp -f $1.data1.EI.good.sorted datafile.dat    
#	cp datafile.dat datafile_EI.dat
#        break
#    fi
#iter=$(($iter + 1))
#done

#cp $1.scan.out.data1.good $1.scan.out.data1.EI.good
#rm -f datafile.dat
#rm -f $1.scan.out.data1.good

#awk 'END {print}' $1.scan.out.data1.EI.good > Last_point_EI
