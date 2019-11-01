#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 2
#$ -o /data/shared/projects/SGE_LOG/

idx=${1} 
n_steps=50000 

hostname
echo ${idx}
echo ${n_steps}

. /home/wiederm/anaconda3/etc/profile.d/conda.sh
conda activate ani36
# nr of jobs: 10080
diameter_in_angstrom=18 #Angstrom
base_path="/data/shared/mwieder/neutromeratio/data/equilibrium_sampling/waterbox-${diameter_in_angstrom}A/${name}/"
cd /home/mwieder/Work/Projects/neutromeratio/scritps
python Generate_equilibrium_sampling_in_droplet.py ${idx} ${n_steps} ${diameter_in_angstrom} ${base_path}
