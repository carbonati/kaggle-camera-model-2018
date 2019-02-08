from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd

N_CLASSES = 10


class CameraDataset(Dataset):
    def __init__(self, path, augmentor, n_tta=11, extend_dataset=False):
        
        self.df = pd.read_csv(path)
        self.files = list(self.df['image_path'])
        self.labels = list(self.df['label'].astype(int))
        self.n_tta = n_tta
        
        self.manip_labels = [-1] * len(self.files)
        
        if extend_dataset:
            _files = self.files
            _labels = self.labels
            
            self.files = []
            self.labels = []
            self.manip_labels = []
            for i in range(len(_files)):
                self.files.extend([_files[i]] * n_tta)
                self.labels.extend([_labels[i]] * n_tta)
                self.manip_labels.extend(list(range(n_tta)))
                
        self.augmentor = augmentor
    
    
    def preprocess(self, index):
        path = self.files[index]
        img = Image.open(path)
        img = np.array(img)
        
        return self.augmentor(img, self.manip_labels[index])
    
    
    def get_labels(self):
        labels = np.zeros((len(self.labels), N_CLASSES))
        labels[range(len(self.labels)), self.labels] = 1
        return labels


    def get_filenames(self):
        filenames = [fp.split('/')[-1] for fp in self.files]
        return filenames
 

    def __getitem__(self, index):
        img = self.preprocess(index)
        label = self.labels[index]
        return img, label
    
    
    def __len__(self):
        return len(self.files)