#! /bin/bash

#BSUB -W 3:00
#BSUB -n 1 
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -o /home/wiederm/LOG/equ_droplet_job%J.log
#BSUB -m "ls-gpu lt-gpu lp-gpu lg-gpu"
#BSUB -L /bin/bash

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
