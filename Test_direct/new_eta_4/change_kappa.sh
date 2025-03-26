#!/bin/bash

sed -i "s/kappa = 2/kappa = $1/" PROGRAM_bayesian_optimization_direct.py
sed -i "s/kappa = 2/kappa = $1/" PROGRAM_bayesian_optimization_direct_test.py
sed -i "s/xi = 2/xi = $1/" PROGRAM_bayesian_optimization_direct.py
sed -i "s/xi = 2/xi = $1/" PROGRAM_bayesian_optimization_direct_test.py
