#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1
￼
molecule_name=$1 
nr_of_run=$2
direction_of_tautomer_transformation=$3
perturbations_per_trial=$4

cd /home/mwieder/Work/Projects/neutromeratio/scripts
hostname
echo ${molecule_name}
echo ${nr_of_run}
echo ${direction_of_tautomer_transformation}
echo ${perturbations_per_trial}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate torchANI-dev

python NCMC.py ${molecule_name} ${nr_of_run} ${direction_of_tautomer_transformation} ${perturbations_per_trial} 'solv'
