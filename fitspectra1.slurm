#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH

srun -n 1 -c 24 python ./fit_spectra.py 
