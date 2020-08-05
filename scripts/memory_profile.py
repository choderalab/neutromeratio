import torchani
from memory_profiler import profile
import torch

@profile
def setup():
    nr_of_methans = 10
    nr_of_frames = 100
    device = 'cpu'
    model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
    coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                                [-0.83140486, 0.39370209, -0.26395324],
                                [-0.66518241, -0.84461308, 0.20759389],
                                [0.45554739, 0.54289633, 0.81170881],
                                [0.66091919, -0.16799635, -0.91037834]]*nr_of_methans]*nr_of_frames,
                            requires_grad=True, device=device)
    species = torch.tensor([[6, 1, 1, 1, 1]*nr_of_methans]*nr_of_frames, device=device)
    energy = model((species, coordinates)).energies

setup()

