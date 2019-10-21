#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -700
#$ -pe smp 2
#$ -o /data/shared/projects/SGE_LOG/
￼#$ -l gpu=1

idx=$1 
n_steps=50000 

cd /home/mwieder/Work/Projects/neutromeratio/scripts
hostname
echo ${idx}
echo ${n_steps}

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate ani36
#mkdir -p ../data/equilibrium_sampling/${molecule_name}
python BFE_Generate_equilibrium_sampling_in_droplet.py ${idx} ${n_steps}
