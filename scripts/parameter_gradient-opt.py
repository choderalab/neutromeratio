import neutromeratio
import pickle
import torch, sys

split = pickle.load(open('split.dict', 'rb'))
names = neutromeratio.parameter_gradients._get_names()

assert(len(sys.argv) == 7)
env = sys.argv[1]
fold = int(sys.argv[2])
elements = sys.argv[3]
data_path = sys.argv[4]
model_name = str(sys.argv[5])
max_snapshots_per_window=int(sys.argv[6])

print(f'Max nr of snapshots: {max_snapshots_per_window}')

if model_name == 'ANI2x':
    model = neutromeratio.ani.AlchemicalANI2x
    print(f'Using {model_name}.')
elif model_name == 'ANI1ccx':
    model = neutromeratio.ani.AlchemicalANI1ccx
    print(f'Using {model_name}.')
elif model_name == 'ANI1x':
    model = neutromeratio.ani.AlchemicalANI1x
    print(f'Using {model_name}.')
else:
    raise RuntimeError(f'Unknown model name: {model_name}')

names_training = [names[i] for i in split[fold][0]]
names_validating = [names[i] for i in split[fold][1]]

assert((len(names_training) + len(names_validating)) == len(names))
assert (11 > fold >= 0)

if env == 'droplet':
    bulk_energy_calculation = False
    torch.set_num_threads(4)
else:
    torch.set_num_threads(4)
    bulk_energy_calculation = True

max_epochs = 0
for _ in range(5):
    max_epochs += 10

    rmse_training, rmse_validation, rmse_test = neutromeratio.parameter_gradients.setup_and_perform_parameter_retraining(
        env=env,
        names_training = names_training,
        names_validating = names_validating,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=max_snapshots_per_window,
        checkpoint_filename= f"parameters_{model_name}_fold_{fold}_{env}.pt",
        data_path=data_path,
        nr_of_nn=8,
        bulk_energy_calculation=bulk_energy_calculation,
        elements=elements,
        max_epochs=max_epochs,
        diameter=18)
    
    f = open(f'results_{model_name}_fold_{fold}_{env}.txt', 'a+')
    
    print('RMSE training')
    print(rmse_training)   
    f.write('RMSE training')
    f.write('\n')

    for e in rmse_training:
        f.write(str(e) + ', ')
    f.write('\n')

    print('RMSE validation')
    f.write('\n')

    print(rmse_validation)
    f.write('RMSE validation')
    f.write('\n')
    for e in rmse_validation:
        f.write(str(e) + ', ')
    f.write('\n')   
    f.close()