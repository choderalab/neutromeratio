import neutromeratio
import pickle
import pkg_resources
from neutromeratio.parameter_gradients import tweak_parameters
from neutromeratio.constants import exclude_set_ANI, mols_with_charge

data = pkg_resources.resource_stream(neutromeratio.__name__, "data/exp_results.pickle")
print(f"data-filename: {data}")
exp_results = pickle.load(data)

names = []
for name in exp_results:
    if name in exclude_set_ANI + mols_with_charge:
        continue
        names.appen(name)

tweak_parameters(data_path='./data', nr_of_nn=8, names = names[:30], max_epochs=100)