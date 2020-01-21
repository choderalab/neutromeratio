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
per_atom_stddev_threshold=10.5
base_path="/data/cluster/projects/neutromeratio/data/equilibrium_sampling/vacuum-stereo/${name}"

echo 'Idx: '${idx}
echo 'Per atom stddev threshold: '${per_atom_stddev_threshold}
echo 'Base path: '${base_path}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36v2
# nr of jobs: 400

python Analyse_equilibrium_samples.py ${idx} ${base_path} ${env} ${per_atom_stddev_threshold}