import torch.utils.data as data
import os
import torch
import tqdm
import imghdr
import random
from PIL import Image
import numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
Image.MAX_IMAGE_PIXELS = 933120000
import random
import cv2
from io import StringIO



mean = [0.42134404, 0.1422577, 0.09813465]
std = [0.48937318, 0.2481444, 0.28232995]

preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)

    #normalize
])


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, recursive_search=False,in_memory: bool = False,):
        super(ImageDataset, self).__init__()

        arraytable = numpy.genfromtxt(data_dir, delimiter=',', dtype=str)

        paths=arraytable[1:,1]
        Cities=arraytable[1:,2]
        PatternTypes=arraytable[1:,5]
        Periods=arraytable[1:,6]

        maxheight=arraytable[1:,8]


        self.npzpaths=paths
        self.target_City=Cities
        self.target_PatternType=PatternTypes
        self.target_Period=Periods
        self.target_height = maxheight

        self.countcountGlobal=0


    def __len__(self):
        return len(self.npzpaths)

    def __getitem__(self, index):

        path=self.npzpaths[index]

        GT_City = self.target_City[index]
        GT_City=GT_City.astype(numpy.float)

        GT_Pattern = self.target_PatternType[index]
        GT_Pattern = GT_Pattern.astype(numpy.float)

        GT_Period = self.target_Period[index]
        GT_Period = GT_Period.astype(numpy.float)

        GT_Maxheight = self.target_height[index]
        GT_Maxheight = GT_Maxheight.astype(numpy.float)

        GT_City_tensor=torch.tensor(GT_City, dtype=torch.int64)
        GT_Pattern_tensor = torch.tensor(GT_Pattern, dtype=torch.int64)
        GT_Period_tensor = torch.tensor(GT_Period, dtype=torch.int64)
        GT_Maxheight_tensor = torch.tensor(GT_Maxheight, dtype=torch.int64)



        self.countcountGlobal = self.countcountGlobal + 1
        image_name='./Samples_png/'+str(path)+'.png'
        image = Image.open(image_name).convert('RGB')
        imagetensor = preprocess(image)
        imagetensor=imagetensor



        item = {}
        item["parcels"] = imagetensor.float()
        item["GT_City"] = GT_City_tensor
        item["GT_Pattern"] = GT_Pattern_tensor
        item["GT_Period"] = GT_Period_tensor
        item["GT_Maxheight"] = GT_Maxheight_tensor
        return item

