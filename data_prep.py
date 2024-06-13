import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Datasets(Dataset):
    """
    Creates the dataset of train and val images
    """
    def __init__(self, path, files):
        super().__init__()
        self.path = path
        self.files = files
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        # returns one sample 
        item = np.load(str(self.path) + str(self.files[idx])) # load item from path
        block = torch.from_numpy(item).type(torch.FloatTensor)
        return block 

class Data():
    def __init__(self, path, buffer, no_timeperiods):
        self.path = path
        self.filenames = self.load_filenames() # loads data into numpy array 
        self.no_timeperiods = no_timeperiods # we will sample 40/40/20 (train, val, test) from each timeperiod

        self.buffer = buffer # number of instances to remove between train/val/test in each timeperiod to prevent weather dependencies 

        # TODO currently is as nparray, want it as tensors? 
        self.train_data, self.val_data, self.test_data = self.test_train_split()

    def load_filenames(self):
        # Load all filenames from directory into a list.
        return listdir(self.path)

    def test_train_split(self):        
        timeperiods = np.array_split(self.filenames, self.no_timeperiods)

        # initiate lists of np arrays
        train_files = [] 
        val_files = []
        test_files = []

        for period in timeperiods: 
            train_val_cutoff = int(round(0.75*len(period)))
            val_test_cutoff = int(round(0.9*len(period)))

            train_files.append(period[self.buffer:train_val_cutoff]) # buffer beginning so periods are independent 
            val_files.append(period[train_val_cutoff+self.buffer : val_test_cutoff])
            test_files.append(period[val_test_cutoff+self.buffer : ])

        # flatten list of lists
        train_files = np.array(train_files).flatten()
        val_files = np.array(val_files).flatten()
        test_files = np.array(test_files).flatten()
        
        # create PyTorch Datasets instances
        train_set = Datasets(self.path, train_files)
        val_set = Datasets(self.path, val_files)
        test_set = Datasets(self.path, test_files)

        return train_set, val_set, test_set