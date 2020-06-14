import neutromeratio
import pickle
import pkg_resources

data = pkg_resources.resource_stream(neutromeratio.__name__, "data/exp_results.pickle")
print(f"data-filename: {data}")
exp_results = pickle.load(data)
env='vacuum'
max_epochs = 0
for _ in range(10):
    max_epochs += 10
    rmse_training, rmse_validation, rmse_test = neutromeratio.parameter_gradients.tweak_parameters(batch_size=10, env=env, data_path='./data/vacuum', nr_of_nn=8, max_epochs=20)   
    f = open('results.txt', 'a+')
    
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
    
    print('RMSE test')
    print(rmse_test)
    f.write('RMSE test')
    for e in rmse_test:
        f.write(str(e) + ', ')
    f.write('\n')
    f.close()