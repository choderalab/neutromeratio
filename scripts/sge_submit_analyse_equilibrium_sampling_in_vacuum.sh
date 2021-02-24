#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/cluster/projects/SGE_LOG/

idx=${1} 
hostname
env='vacuum'
potential_name='ANI1ccx'
echo 'Using potential ' ${potential_name}
base_path="/data/shared/projects/neutromeratio/data/equilibrium_sampling/${potential_name}_${env}-200ps-p2"
echo 'Idx: '${idx}
echo 'Base path: '${base_path}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani
# nr of jobs: 400
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Analyse_equilibrium_samples.py ${idx} ${base_path} ${env} ${potential_name}