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

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36
# nr of jobs: 10080
base_path="/data/shared/projects/neutromeratio/data/equilibrium_sampling/vacuum/${name}"
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Generate_equilibrium_sampling_in_vacuum.py ${idx} ${n_steps} ${base_path}
