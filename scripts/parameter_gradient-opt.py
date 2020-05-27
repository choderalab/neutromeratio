import neutromeratio
import pickle
import pkg_resources

data = pkg_resources.resource_stream(neutromeratio.__name__, "data/exp_results.pickle")
print(f"data-filename: {data}")
exp_results = pickle.load(data)

rmse_training, rmse_validation, rmse_test = neutromeratio.parameter_gradients.tweak_parameters(batch_size=10, data_path='./data', nr_of_nn=8, max_epochs=20)
print('RMSE training')
print(rmse_training)
print('RMSE validation')
print(rmse_validation)
print('RMSE test set')
print(rmse_test)
