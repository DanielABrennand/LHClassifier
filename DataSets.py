from torch.utils.data import Dataset
from torch import is_tensor
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os
import h5py

class H5FrameDataSet(Dataset):
    def __init__(self,root,csv_path,image_path,transform=None):
        self.root = root
        self.image_path = image_path
        self.csv_path = csv_path
        self.Frames = pd.read_csv(os.path.join(root,csv_path),header = None)
        self.transform = transform

    def __len__(self):
        return len(self.Frames)

    def __getitem__(self,idx):
        if is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_path,(self.Frames.iloc[idx,0] + '.jpg'))
       
        image = read_image(os.path.join(self.root,img_name))

        #image = Image.open(os.path.join(self.root,img_name))
        #image = io.imread(os.path.join(self.root,img_name))

        mode = self.Frames.iloc[idx,1]

        if self.transform:
            image = self.transform(image)

        sample = {'image' : image, 'mode' : mode}

        return sample


class NumpyDataSet(Dataset):
    def __init__(self,image_path,mode_path,transform=None):
        self.Modes = np.load(mode_path)
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.Modes)

    def __getitem__(self, index):
        mode = self.Modes[index]

        image = np.load("{}/{}.npy".format(self.image_path,str(index).zfill(4)))
        if self.transform:
            image = self.transform(image)

        sample = {'xx' : image, 'yy' : mode}
        return sample

class H5DataSet(Dataset):
    def __init__(self,h5path,tagpath):
        self.h5path = h5path
        self.files = os.listdir(h5path)
        self.files.sort()
        self.total = 0
        self.dividers = [0]
        for file in self.files:
            self.total += len(h5py.File(os.path.join(h5path,file)).keys())-1
            self.dividers.append(self.total)

        self.tags = pd.read_csv(tagpath,header = None)

    def __len__(self):
        return self.total

    def __getitem__(self,idx):

        filefrom = ""
        for n,divider in enumerate(self.dividers):
            if idx < divider:
                filefrom = self.files[n-1]
                index = idx-self.dividers[n-1]
                break
            
        framename = "frame{}".format(str(index).zfill(4))

        image = np.array(h5py.File(os.path.join(self.h5path,filefrom))[framename])

        mode = self.tags[idx]

        sample = {'image' : image, 'mode' : mode}
        return sample
