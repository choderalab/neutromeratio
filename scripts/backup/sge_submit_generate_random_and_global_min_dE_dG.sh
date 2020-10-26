#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/cluster/projects/SGE_LOG/

idx=${1} 

hostname
echo ${idx}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v2
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Generate_random_and_global_min_dE_dG.py ${idx} 
