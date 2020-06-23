#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/cluster/projects/SGE_LOG/

idx=${1} 
n_steps=10000 
env='droplet'
potential_name='ANI1ccx'

hostname
echo ${idx}
echo ${n_steps}
echo ${env}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v3
# nr of jobs: 10080
diameter_in_angstrom=18 #Angstrom
base_path="/data/shared/projects/neutromeratio/data/equilibrium_sampling/${potential_name}waterbox-${diameter_in_angstrom}A/${name}"
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Generate_equilibrium_sampling.py ${idx} ${n_steps} ${base_path} ${env} ${diameter_in_angstrom} 
