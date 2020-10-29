########################################
# idx has to be provided
idx=${1}
########################################
# idx is an integer between 0 and 5000
# idx is used to index in this code:

#names = neutromeratio.parameter_gradients._get_names()
#for name in names:
#    for lamb in np.linspace(0, 1, 11):
#        protocol.append((name, np.round(lamb, 2)))
#name, lambda_value = protocol[idx-1]
########################################

n_steps=400000 # nr of steps 
env='vacuum'
potential_name='ANI1ccx' # which potential do you want to use? (ANI1ccx, ANI1x, ANI2x)
echo 'Using potential ' ${potential_name}

hostname
echo ${idx}
echo ${n_steps}
echo ${env}

base_path="./" # where do you want to save the ouput files
mkdir -p ${base_path}

python Generate_equilibrium_samples.py ${idx} ${n_steps} ${base_path} ${env} ${potential_name}