import neutromeratio
import pickle
import sys
import torch

assert(len(sys.argv) == 5)
env = sys.argv[1]
elements = sys.argv[2]
data_path = sys.argv[3]
model_name = str(sys.argv[4])

max_snapshots_per_window=200
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


if env == 'droplet':
    bulk_energy_calculation = False
    torch.set_num_threads(4)
else:
    torch.set_num_threads(1)
    bulk_energy_calculation = True

max_epochs = 0
for _ in range(5):
    max_epochs += 10

    rmse_training, rmse_validation = neutromeratio.parameter_gradients.setup_and_perform_parameter_retraining_with_test_set_split(
        env=env,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=max_snapshots_per_window,
        checkpoint_filename= f"parameters_{model_name}_{env}.pt",
        data_path=data_path,
        nr_of_nn=8,
        bulk_energy_calculation=bulk_energy_calculation,
        elements=elements,
        max_epochs=max_epochs,
        diameter=18)
    
    f = open(f'results_{model_name}_{env}.txt', 'a+')
    
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