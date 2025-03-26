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
qp set cap eta_cap 0.00
qp set cap eta_step_size 0.001
qp set cap n_steps_cap  30
qp set davidson_keywords n_det_max_full 1000
qp set davidson_keywords state_following false
qp set determinants read_wf True
#qp set determinants n_states 3

qp run diagonalize_h_cap > $1.scan.test.out


