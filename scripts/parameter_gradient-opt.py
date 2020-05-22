import neutromeratio
import pickle
import pkg_resources

data = pkg_resources.resource_stream(neutromeratio.__name__, "data/exp_results.pickle")
print(f"data-filename: {data}")
exp_results = pickle.load(data)

names = []
for name in exp_results:
    if name in neutromeratio.constants.exclude_set_ANI + neutromeratio.constants.mols_with_charge:
        continue
    names.append(name)

neutromeratio.parameter_gradients.tweak_parameters(batch_size=10, data_path='./data', nr_of_nn=8, max_epochs=100)
