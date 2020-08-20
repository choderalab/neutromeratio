#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/shared/projects/SGE_LOG/

idx=${1} 
n_steps=400000 
nv='vacuum'
potential_name='ANI1ccx'
echo 'Using potential ' ${potential_name}

hostname
echo ${idx}
echo ${n_steps}
echo ${env}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v3
# nr of jobs: 400
base_path="/data/shared/projects/neutromeratio/data/equilibrium_sampling/${potential_name}_${env}-200ps-p5/"
mkdir -p ${base_path}
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Generate_equilibrium_sampling.py ${idx} ${n_steps} ${base_path} ${env} ${potential_name}