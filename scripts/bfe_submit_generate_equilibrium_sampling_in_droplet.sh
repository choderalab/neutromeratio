#! /bin/bash

#BSUB -W 18:00
#BSUB -n 2
#BSUB -R span[ptile=2]
#BSUB -o /home/wiederm/LOG/equ_droplet_job%J.log
#BSUB -L /bin/bash
#BSUB -R rusage[mem=1]

idx=$LSB_JOBINDEX 
n_steps=50000 

cd /home/wiederm/neutromeratio/neutromeratio/scripts
hostname
echo ${idx}
echo ${n_steps}

. /home/wiederm/anaconda3/etc/profile.d/conda.sh
conda activate ani36
#mkdir -p ../data/equilibrium_sampling/${molecule_name}
python BFE_Generate_equilibrium_sampling_in_droplet.py ${idx} ${n_steps}
