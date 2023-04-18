from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
from pandas import DataFrame
import numpy as np

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data:DataFrame, mode:str):
        super().__init__()
        self.data = data
        self.mode = mode
        self.train_transforms=[tv.transforms.ToPILImage(), 
                               tv.transforms.RandomVerticalFlip(p=0.5),
                               tv.transforms.RandomHorizontalFlip(p=0.5),
                               #tv.transforms.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02)),
                               #tv.transforms.RandomResizedCrop((300, 300), scale=(0.98, 1.0), ratio=(1.0, 1.0)),
                               tv.transforms.ToTensor(), 
                               tv.transforms.Normalize(train_mean, train_std)]
        self.val_transforms=[tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)]
        self._transform = tv.transforms.Compose(transforms=self.val_transforms)
    # TODO implement the Dataset class according to the description
    #pass
    #tv.transforms.RandomChoice([tv.transforms.RandomRotation((90,90)), tv.transforms.RandomRotation((-90,-90))],p=(0.5,0.5))

    @property
    def transform(self):
        return self._transform
    @transform.setter
    def transform(self, transforms_list):
        self._transform = tv.transforms.Compose(transforms=transforms_list if transforms_list is not None else self.val_transforms)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #filename, isCrack, isInactive = self.data.at[index, "filename"],self.data.at[index, "crack"], self.data.at[index, "inactive"]
        filename, isCrack, isInactive = self.data.iloc[index]   #when not using stratify_labels
        img = imread(Path(filename))
        img = gray2rgb(img)
        img=torch.from_numpy(np.transpose(img, (2, 0, 1)))
        if self.mode=='train':
            self.transform=self.train_transforms
            transformer = self.transform
            '''if random.random() < 0.5:
                img = tv.transforms.functional.rotate(img,90)'''
        else:
            self.transform=self.val_transforms
            transformer = self.transform
        img = transformer(img)
        return (img, torch.tensor([isCrack, isInactive]))
    
    def calc_class_weight(self):
        epsilon=1e-15
        len=self.__len__()
        sum_crack=0
        sum_inactive=0
        for i in range(len):
            #isCrack, isInactive = self.data.at[i, "crack"], self.data.at[i, "inactive"]
            _, isCrack, isInactive = self.data.iloc[i]
            sum_crack += int(isCrack)
            sum_inactive += int(isInactive)
        w_crack = torch.tensor((len - sum_crack) / (sum_crack + epsilon))
        w_inactive = torch.tensor((len - sum_inactive) / (sum_inactive + epsilon))
        classWeight_tensor = torch.zeros((2))
        classWeight_tensor[0] = w_crack
        classWeight_tensor[1] = w_inactive
        return classWeight_tensor
    
    def getfile(self, filename:str, random=False):
        if not random:
            len=self.__len__()
            for i in range(len):
                #location, isCrack, isInactive = self.data.at[i, "filename"], self.data.at[i, "crack"], self.data.at[i, "inactive"]
                location, isCrack, isInactive = self.data.iloc[i]
                if str(location)==filename:
                    img = imread(Path(str(location)))
                    img_transformed = gray2rgb(img)
                    self.transform=self.val_transforms
                    transformer = self.transform
                    img_transformed = transformer(img_transformed)
                    return (img, img_transformed, torch.tensor([isCrack, isInactive]))
        else:
            i=torch.randint(0,self.__len__(),(1,))
            #location, isCrack, isInactive = self.data.at[i[0], "filename"], self.data.at[i[0], "crack"], self.data.at[i[0], "inactive"]
            location, isCrack, isInactive = self.data.iloc[i[0]]
            img = imread(Path(location))
            img_transformed = gray2rgb(img)
            self.transform=self.val_transforms
            transformer = self.transform
            img_transformed = transformer(img_transformed)
            return (img, img_transformed, torch.tensor([isCrack, isInactive]))