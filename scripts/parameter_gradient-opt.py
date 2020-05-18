import neutromeratio
import pickle
import pkg_resources
from neutromeratio.parameter_gradients import tweak_parameters


data = pkg_resources.resource_stream(neutromeratio.__name__, "data/exp_results.pickle")
print(f"data-filename: {data}")
exp_results = pickle.load(data)
names = [name for name in exp_results][:20] 
tweak_parameters(data_path='./data', nr_of_nn=8, names = ['SAMPLmol2', 'SAMPLmol4'], max_epochs=100)