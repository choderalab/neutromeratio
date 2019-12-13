#! /bin/bash

#BSUB -W 28:00
#BSUB -n 1
#BSUB -o /home/wiederm/LOG/equ_droplet_job%J.log
#BSUB -L /bin/bash
#BSUB -R rusage[mem=1]

idx=$LSB_JOBINDEX 
n_steps=50000 
per_atom_stddev_threshold=0.5
hostname
echo ${idx}
echo ${n_steps}
echo ${per_atom_stddev_threshold}

. /home/wiederm/anaconda3/etc/profile.d/conda.sh
conda activate ani36
# nr of jobs: 10080
diameter_in_angstrom=18 #Angstrom
base_path="/data/chodera/wiederm/equilibrium_sampling/waterbox-${diameter_in_angstrom}A/${name}/"

python Generate_equilibrium_sampling_in_droplet.py ${idx} ${n_steps} ${diameter_in_angstrom} ${base_path}
