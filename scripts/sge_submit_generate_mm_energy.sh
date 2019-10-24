#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 2
#$ -o /data/shared/projects/SGE_LOG/
￼
export NUMEXPR_MAX_THREADS=2
molecule_name=$1 

cd /home/mwieder/Work/Projects/neutromeratio/scripts
hostname
echo ${molecule_name}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36

python Generate_MM_energy.py ${molecule_name}
