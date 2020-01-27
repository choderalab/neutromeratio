#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 1
#$ -o /data/cluster/projects/SGE_LOG/

idx=${1} 
n_steps=100000 

hostname
echo ${idx}
echo ${n_steps}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v2
# nr of jobs: 400
base_path="/data/cluster/projects/neutromeratio/data/equilibrium_sampling/vacuum-stereo/${name}"
cd /home/mwieder/Work/Projects/neutromeratio/scripts
python Generate_equilibrium_sampling.py ${idx} ${n_steps} ${base_path} "vacuum"
