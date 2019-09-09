#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 3
#$ -o /data/shared/projects/SGE_LOG/
￼
molecule_name=$1 
lambda_value=$2
n_steps=50000

cd /home/mwieder/Work/Projects/neutromeratio/scripts
hostname
echo ${molecule_name}
echo ${lambda_value}
echo ${n_steps}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate torchANI-dev

python Equilibrium_sampling.py ${molecule_name} ${lambda_value} 'cpu' ${n_steps}
