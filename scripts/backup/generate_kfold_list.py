import pickle
import neutromeratio
from sklearn.model_selection import KFold
import random

names = neutromeratio.parameter_gradients._get_names()
splits = {}
kf = KFold(n_splits=10, shuffle=True)
for idx, (train, test) in enumerate(kf.split(names)):
    splits[idx] = [train, test]

f = open('split.dict', 'wb+')
pickle.dump(splits, f)
print(splits)