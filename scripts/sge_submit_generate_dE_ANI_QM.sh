#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 2
#$ -o /data/cluster/projects/SGE_LOG/

idx=${1} 
hostname
pid=$$

echo 'Idx: '${idx}

mkdir -p /data/local/psi4-scratch/
mkdir -p /data/local/psi4-${pid}
cd /data/local/psi4-${pid}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v2
# nr of jobs: 400
python /home/mwieder/Work/Projects/neutromeratio/scripts/Generate_dE_ANI_QM.py ${idx} ANI1x 

rm -r /data/local/psi4-${pid}
