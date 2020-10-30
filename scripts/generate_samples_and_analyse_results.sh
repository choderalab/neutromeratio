# This is a simple wrapper script that generates equilibrium samples for a given tautomer set 
# and estimates the free energy difference with ANI1cccx and the tautomer optimized parameter set
################################################
SMILES1='OC1=CC=C2C=CC=CC2=N1'
SMILES2='O=C1NC2=C(C=CC=C2)C=C1'
name='test_mol' # defines where the output directory name
base_path="./" # where do you want to save the ouput files -> the ouput will have the form ${base_path}/${name}
potential_name='ANI1ccx' # which potential do you want to use? (ANI1ccx, ANI1x, ANI2x)
n_steps=10000 # nr of steps (dt = 0.5fs)
env='vacuum' 
################################################
echo 'Using potential ' ${potential_name}
echo 'Nr of steps : ' ${n_steps}
echo 'Simulating in : ' ${env}
echo 'Lambda value: ' ${lambda_value}
mkdir -p ${base_path}

for lambda_value in $(seq 0 0.2 1); # using 6 lambda states
do python Generate_equilibrium_samples_for_new_tautomer_pairs.py ${SMILES1} ${SMILES2} ${name} ${lambda_value} ${n_steps} ${base_path} ${env} ${potential_name};
done

python Calculate_dG_for_new_tautomer_pairs.py ${SMILES1} ${SMILES2} ${name} ${base_path} ${env} ${potential_name};
