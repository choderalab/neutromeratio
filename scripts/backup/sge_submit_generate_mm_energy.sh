#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 2
#$ -o /data/shared/projects/SGE_LOG/
￼
export NUMEXPR_MAX_THREADS=2
idx=$1 

cd /home/mwieder/Work/Projects/neutromeratio/scripts
hostname

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v2

python Generate_MiningMinima_energy.py ${idx}
