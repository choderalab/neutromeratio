
# User specified runtime variables
########################################
SMILES1='OC1=CC=C2C=CC=CC2=N1'
SMILES2='O=C1NC2=C(C=CC=C2)C=C1'
lambda_value=0.0   # alchemical coupling parameter (between 0. and 1.) --> we recommend using 11 lambda_values
name='SAMP2Lmol4'
potential_name='ANI1ccx' # which potential do you want to use? (ANI1ccx, ANI1x, ANI2x)
n_steps=200 # nr of steps (dt = 0.5fs)
########################################
########################################


env='vacuum' 
echo 'Using potential ' ${potential_name}
echo 'Nr of steps : ' ${n_steps}
echo 'Simulating in : ' ${env}
echo 'Lambda value: ' ${lambda_value}

base_path="./" # where do you want to save the ouput files
mkdir -p ${base_path}

python Generate_equilibrium_samples_for_new_tautomer_pairs.py ${SMILES1} ${SMILES2} ${name} ${lambda_value} ${n_steps} ${base_path} ${env} ${potential_name}