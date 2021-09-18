from torch.utils.data import Dataset
from data.example_bin import load_example
import glob
import torch
import tqdm
import numpy as np
from .transforms import BatchwiseStandardizeTransform, ElementwiseScalarStandardizeTransform
from typing import Tuple


class PANNADiamineDataset(Dataset):
    def __init__(self, data_directory='data/nograd_bulk', standardize_x=False, standardize_y=False):
        super(PANNADiamineDataset, self).__init__()
        self.filenames = glob.glob(data_directory + '/*')
        # energies obtained from pseudopotentials
        # indexed as 0=hydrogen, 1=C, 2=N, 3=O, 4=Mg
        self.atomic_energy = np.array([-12.741175746224702, -245.6676073698599, -380.56317773304966, -565.9492355766356, -1932.5893968794142])

        self.standardize_x = standardize_x
        self.standardize_y = standardize_y
        if self.standardize_y:
            self.transform_y = ElementwiseScalarStandardizeTransform()
        if self.standardize_x:
            print('Finding mean and variance for standardization procedure...')
            self.transform_x = BatchwiseStandardizeTransform()
            for i in tqdm.trange(len(self)):
                item = self._get_raw(i)
                self.transform_x.update(item[0])
                if self.standardize_y:
                    self.transform_y.update(item[2])
        if self.standardize_x is False and self.standardize_y:
            for i in tqdm.trange(len(self)):
                self.transform_y.update(self._get_raw(i)[2])

    def __len__(self):
        return len(self.filenames)

    def _get_raw(self, item: int) -> Tuple[torch.FloatTensor, torch.LongTensor, float]:
        filename = self.filenames[item]
        ex = load_example(filename)
        cohesive_energy = ex.true_energy - self.atomic_energy[ex.species_vector].sum()
        return torch.from_numpy(ex.gvects).float(), torch.from_numpy(ex.species_vector), cohesive_energy

    def __getitem__(self, item: int):
        x, species_idx, cohesive_energy = self._get_raw(item)
        if self.standardize_x:
            x = self.transform_x(x)

        if self.standardize_y:
            cohesive_energy = self.transform_y(cohesive_energy)
        return x, species_idx, cohesive_energy


class InMemoryPANNA(PANNADiamineDataset):
    def __init__(self, data_directory='data/nograd_bulk', standardize_x=False, standardize_y=False):
        super(InMemoryPANNA, self).__init__(data_directory, standardize_x, standardize_y)
        self._cache = [super(InMemoryPANNA, self).__getitem__(i) for i in tqdm.trange(self.__len__(), desc='Caching data into memory... ')]

    def __getitem__(self, item):
        return self._cache[item]
