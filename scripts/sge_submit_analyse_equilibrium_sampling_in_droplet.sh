#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/cluster/projects/SGE_LOG/

idx=${1} 
hostname
env='droplet'
diameter=18
potential_name='ANI2x'
echo 'Using potential ' ${potential_name}
base_path="/data/shared/projects/neutromeratio/data/equilibrium_sampling/${potential_name}-${env}-${diameter}A-100ps"

echo 'Idx: '${idx}
echo 'Base path: '${base_path}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v3
# nr of jobs: 400
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Analyse_equilibrium_samples.py ${idx} ${base_path} ${env} ${potential_name}
