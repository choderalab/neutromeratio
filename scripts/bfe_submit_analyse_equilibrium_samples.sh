#! /bin/bash

#BSUB -W 48:00
#BSUB -n 2
#BSUB -R span[ptile=2]
#BSUB -o /home/wiederm/LOG/equ_droplet_job%J.log
#BSUB -L /bin/bash
#BSUB -R rusage[mem=3]

idx=$LSB_JOBINDEX
hostname
per_atom_stddev_treshold=0.5
diameter_in_angstrom=18 #Angstrom
base_path="/data/chodera/wiederm/equilibrium_sampling/waterbox-${diameter_in_angstrom}A/"
env='droplet'

echo 'Idx: '${idx}
echo 'Per atom stddev treshold: '${per_atom_stddev_treshold}
echo 'Diameter in Angstrom: '${diameter_in_angstrom}
echo 'Base path: '${base_path}

. /home/wiederm/anaconda3/etc/profile.d/conda.sh
conda activate ani36
# nr of jobs: 500

python Analyse_equilibrium_samples_in_droplet.py ${idx} ${base_path} ${env} ${per_atom_stddev_treshold} ${diameter_in_angstrom}
