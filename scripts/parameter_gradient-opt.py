import neutromeratio
import pickle
import sys

split = pickle.load(open('split.dict', 'rb'))
names = neutromeratio.parameter_gradients._get_names()
model_name = 'AlchemicalANI1ccx'
assert(len(sys.argv) == 3)
env = sys.argv[1]
fold = int(sys.argv[2])

names_training = [names[i] for i in split[fold][0]]
names_validating = [names[i] for i in split[fold][1]]

assert((len(names_training) + len(names_validating)) == len(names))
assert (11 > fold > 0)

for n in ['molDWRow_1636', 'molDWRow_1250', 'molDWRow_1228']:
    if n in names_training:
        names_training.remove(n)
    if n in names_validating:
        names_validating.remove(n)


max_epochs = 0
for _ in range(5):
    max_epochs += 10

    rmse_training, rmse_validation = neutromeratio.parameter_gradients.tweak_parameters_for_list(
        env=env,
        names_training = names_training,
        names_validating = names_validating,
        ANImodel=neutromeratio.ani.AlchemicalANI1ccx,
        batch_size=1,
        max_snapshots_per_window=100,
        checkpoint_filename= f"parameters_{model_name}_fold_{fold}_{env}.pt",
        data_path=f'/data/shared/projects/neutromeratio/data/equilibrium_sampling/waterbox-18A-V2/',
        nr_of_nn=8,
        elements='CHON',
        max_epochs=max_epochs,
        diameter=18)
    
    f = open(f'results_{model_name}_fold_{fold}_{env}.txt', 'a+')
    
    print('RMSE training')
    print(rmse_training)   
    f.write('RMSE training')
    for e in rmse_training:
        f.write(str(e) + ', ')
    f.write('\n')

    print('RMSE validation')
    print(rmse_validation)
    f.write('RMSE validation')
    for e in rmse_validation:
        f.write(str(e) + ', ')
    f.write('\n')   
    f.close()