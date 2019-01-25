from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd

N_CLASSES = 10
TTA_COUNT = 11


class CameraDataset(Dataset):
    def __init__(self, path, augmentor, extend_dataset=False):        
        self.df = pd.read_csv(path)
        self.df = self.df.iloc[0:35]
        self.files = list(self.df['image_path'])
        self.labels = list(self.df['label'].astype(int))
        
        self.manip_labels = [-1] * len(self.files)
        
        if extend_dataset:
            _files = self.files
            _labels = self.labels
            
            self.files = []
            self.labels = []
            self.manip_labels = []
            for i in range(len(_files)):
                self.files.extend([_files[i]] * TTA_COUNT)
                self.labels.extend([_labels[i]] * TTA_COUNT)
                self.manip_labels.extend(list(range(TTA_COUNT)))
                
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
 

    def __getitem__(self, index):
        img = self.preprocess(index)
        label = self.labels[index]
#         label = np.zeros(N_CLASSES)
#         label[self.labels[index]] = 1
        return img, label
    
    
    def __len__(self):
        return len(self.files)