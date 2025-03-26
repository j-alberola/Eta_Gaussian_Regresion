#!/usr/bin/env qpsh

#SBATCH -J bu.p
#SBATCH -p zen3 -c 30  -n 1 -N 1
#SBATCH --mem=20000


# Set name of the molecule and basis set
#INPUT='N2_minus'
#BASIS=basis_N2_aug-cc-pvtz-3s3p3d


# Source QP2
module load intel
module load lcpq/gnu/gcc-7.5.0
source ~/PHD/qp2/quantum_package.rc


#LAUNCH DESIRED CALCUALTION WITH SPECIFIED PARAMETERS
#
##MODIFIY FORTRAN SCRIPT TO WRITE OUTPUT IN A GIVEN FILE

qp set_file $1
#qp edit -s [1,2]
qp set cap do_cap true
qp set cap eta_cap 0.001
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
python3.11 PROGRAM_bayesian_optimization_direct_test.py 

threshold_eta=0.00001
threshold_velocity=0.00001
qp set cap eta_step_size 0.
qp set cap n_steps_cap  1



#  ADD A LOOP HERE
#  MAYBE REMOVE THE DIFFERENT 2 CASES, JUST 1 IS ENOUGH

iter=1
while [ $iter -lt 7 ] ; do


    new_min=$(awk 'NR == 1 {print $1}' Optimal_LB)
    qp set cap eta_cap $new_min
    qp run diagonalize_h_cap > $1.scan.LB.out
    cp $1.scan.LB.out $1.scan.script.$new_min.LB.out
    ./script_data_corrected.sh $1.scan.LB.out 1 2
    echo $new_min > min_file
    awk {'print $2, $3, $4, $5, $6, $7, $8'} $1.scan.LB.out.data2.good > energy_file
    paste min_file energy_file >> $1.scan.out.data1.good
    sort -g -k1,1 $1.scan.out.data1.good > $1.data1.LB.good.sorted
    \cp -f $1.data1.LB.good.sorted datafile.dat
    python3.11 PROGRAM_bayesian_optimization_direct_test.py


iter=$(($iter + 1))
done


