#!/usr/bin/env qpsh

#SBATCH -J bu.p
#SBATCH -p multi -c 48  -n 1 -N 1
#SBATCH --mem=200000


# Set name of the molecule and basis set
#INPUT='N2_minus'
#BASIS=basis_N2_aug-cc-pvtz-3s3p3d


# Source QP2
module load intel
module load lcpq/gnu/gcc-7.5.0
source ~/PHD/qp2/quantum_package.rc

for i in {1..10}; do

    mkdir version3_newest_eta$i
    cp -r N2_minus.hf.4states.no.cipsi.17200 version3_newest_eta$i
    cp Good_columns.py version3_newest_eta$i
    cp PROGRAM* version3_newest_eta$i
    cp datafile_big_scan.dat version3_newest_eta$i
    cp change_kappa2.sh version3_newest_eta$i
    cp script* version3_newest_eta$i

    cd version3_newest_eta$i
    ./change_kappa2.sh $i

    ./script_scan_version_3.sh N2_minus.hf.4states.no.cipsi.17200

#    awk -v i="$i" 'END {print "0.001_0.0085", i, "EI", $0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.EI.good >> ../CONVERGED_MINIMA
#    awk -v i="$i" 'END {print "0.000_0.0085", i, "LB", $0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.LB.good >> ../CONVERGED_MINIMA
    awk -v i="$i" 'END {print "0.000_0.0085", i, "LB", $0}' Last_point_LB  >> ../CONVERGED_MINIMA_NEW_3

    for j in 0.01 0.015 0.02 0.03 0.05; do
        mkdir $j
        cp -r N2_minus.hf.4states.no.cipsi.17200 $j
        cp Good_columns.py $j
        cp PROGRAM* $j
        cp datafile_big_scan.dat $j
        cp script* $j

        cd $j

#        sed -i "s/qp set cap eta_cap 0.001/qp set cap eta_cap 0.000/" script_scan_version_2.sh
        sed -i "s/qp set cap eta_step_size 0.0075/qp set cap eta_step_size $j/" script_scan_version_3.sh
        ./script_scan_version_3.sh N2_minus.hf.4states.no.cipsi.17200

#        awk -v j="$j" -v i="$i" 'END {print "0.000_" j, i, "EI", $0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.EI.good >> ../../CONVERGED_MINIMA
        awk -v j="$j" -v i="$i" 'END {print "0.000_" j, i, "LB", $0}' Last_point_LB >> ../../CONVERGED_MINIMA_NEW_3

        cd ..
    done

    cd ..
done


#for i in {1..9}; do
#	
#mkdir newest_eta$i
#cp -r N2_minus.hf.4states.no.cipsi.17200 newest_eta$i
#cp Good_columns.py newest_eta$i
#cp PROGRAM* newest_eta$i
#cp datafile_big_scan.dat newest_eta$i
#cp change_kappa.sh newest_eta$i
#cp script* newest_eta$i
#
#cd newest_eta$i
#./change_kappa.sh $i
#sbatch script_scan_version_2.sh N2_minus.hf.4states.no.cipsi.17200
#
#awk 'END {print "0.001_0.0085", "'"$i"'", "EI",$0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.EI.good >> ../CONVERGED_MINIMA
#awk 'END {print "0.001_0.0085", "'"$i"'", "LB",$0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.LB.good >> ../CONVERGED_MINIMA
#
#for j in 0.0085 0.01 0.012 0.015 0.02 0.025 0.03;do
#mkdir $j
#cp -r N2_minus.hf.4states.no.cipsi.17200 $j
#cp Good_columns.py $j
#cp PROGRAM* $j
#cp datafile_big_scan.dat $j
#cp script* $j
#
#cd $j
#
#sed -i "s/qp set cap eta_cap 0.001/qp set cap eta_cap 0.000/" script_scan_version_2.sh
#sed -i "s/qp set cap eta_step_size 0.0075/qp set cap eta_step_size $j/" script_scan_version_2.sh
#sbatch script_scan_version_2.sh N2_minus.hf.4states.no.cipsi.17200
##sed -i "s/qp set cap eta_cap 0.001/qp set cap eta_cap 0.000/" script_scan_version_2.sh
##sed -i "s/./script_data_corrected.sh $1.scan.out 2 1/./script_data_corrected.sh $1.scan.out 2 1/" script_scan_version_2.sh
#awk 'END {print "0.000_"'$j'"", "'"$i"'", "EI",$0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.EI.good >> ../../CONVERGED_MINIMA
#awk 'END {print "0.000_"'$j'"", "'"$i"'", "LB",$0}' N2_minus.hf.4states.no.cipsi.17200.scan.out.data1.LB.good >> ../../CONVERGED_MINIMA
#
#cd ..
#done
#
#cd ..
#
#done
